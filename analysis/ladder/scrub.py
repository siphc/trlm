"""P3, mechanistic faithfulness (logs/ladder/design.md; decision 17): the swap test (causal scrubbing, Chan et al.,
Redwood Research, 2022) for key claims and the replacement test for value claims.

An account states its claims as a module-level list
    CLAIMS = [{"name": ..., "site": ..., "key": f}, {"name": ..., "site": ..., "value": g}, ...]
where site is one of the places that write to the residual stream of a one-block attention-only model:
    "direct"     the embedding x0 = wte[x_t] + wpe[t] as it enters the final sum (attention still reads the true x0)
    "head<h>"    head h's output z_h at the position (before the output projection)
Exactly one claim per site; every site is claimed.
- key claim: f(X) maps the input windows X (n, T) to the key per position, (n, T) or (n, T, k) integers; inputs with
  the same key are, by the claim, the same for that site. Swap test: each position's site value is replaced by its
  value at a donor position, a random other (window, position) of dev_full with the same key. A position whose key
  value no other position shares keeps its own value; if more than 1% of positions lack a donor, the claim fails.
- value claim: g(X) returns the site's activation itself, (n, T, d) for "direct" or (n, T, dh) for a head, computed by
  the account. Replacement test: the site's value is replaced by g(X).
The model is then finished as usual (residual sum, final LayerNorm, unembedding). The resample ablation draws the
donor with no constraint.
    faithfulness = 1 - KL(model || patched) / KL(model || resample-ablated)
Pass: >= 0.95 with all claims applied together and for each claim alone. Means over DRAWS donor draws.

    .venv/bin/python -m analysis.ladder.scrub --rung L1 --arm ln --seed 0 --account ladder/L1_ln/account.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analysis.ladder.common import kl_pos, load, load_account, model_logprobs, windows

MIN_FAITHFULNESS = 0.95
MAX_NO_DONOR = 0.01
DRAWS = 3


@torch.no_grad()
def parts(model, X: np.ndarray):
    """x0 (n, T, d) and z (n, T, H, dh) as float64 arrays (the model is in float64)."""
    x = torch.as_tensor(X, dtype=torch.int64)
    n, T = x.shape
    blk = model.blocks[0]
    H, dh, d = model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(d, dim=-1)
    q, k, v = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k, v))
    z = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    return x0.double().numpy(), z.double().numpy()


@torch.no_grad()
def finish(model, x0: np.ndarray, z: np.ndarray) -> np.ndarray:
    n, T = x0.shape[:2]
    h = torch.as_tensor(x0) + model.blocks[0].attn.out(torch.as_tensor(z).reshape(n, T, -1))
    return torch.log_softmax(model.ln_f(h) @ model.wte.weight.T, -1).numpy()


def key_ids(key: np.ndarray) -> np.ndarray:
    """Flatten a key array (n, T) or (n, T, k) to one integer id per position."""
    k = np.asarray(key)
    k = k.reshape(k.shape[0] * k.shape[1], -1)
    return np.unique(k, axis=0, return_inverse=True)[1].ravel()


def donors(ids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random other position with the same id for each position (cyclic shift inside a shuffled group), and a mask
    of positions that have one."""
    N = len(ids)
    order = np.lexsort((rng.random(N), ids))
    ks = ids[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    size = np.diff(np.r_[start, N])
    g = np.repeat(np.arange(len(start)), size)
    p = np.arange(N) - start[g]
    donor_sorted = start[g] + (p + 1) % size[g]
    d = np.empty(N, np.int64)
    d[order] = order[donor_sorted]
    return d, d != np.arange(N)


def scrubbed(model, x0, z, sites: dict, rng) -> np.ndarray:
    """Finish the model with each site in `sites` patched: site -> ("key", ids) swap test, ("value", array)
    replacement, or None for an unconstrained donor (the resample ablation)."""
    n, T = x0.shape[:2]
    N = n * T
    x0d, zd = x0.reshape(N, -1).copy(), z.reshape(N, z.shape[2], -1).copy()
    for site, how in sites.items():
        if how is not None and how[0] == "value":
            v = np.asarray(how[1], np.float64).reshape(N, -1)
            if site == "direct":
                x0d = v
            else:
                zd[:, int(site[4:])] = v
            continue
        d = rng.permutation(N) if how is None else donors(how[1], rng)[0]
        if site == "direct":
            x0d = x0d[d]
        else:
            h = int(site[4:])
            zd[:, h] = zd[d, h]
    return finish(model, x0d.reshape(x0.shape), zd.reshape(z.shape))


def run(rung: str, arm: str, seed: int, account: str, split: str = "dev_full") -> dict:
    model = load(rung, arm, seed).double()
    W = windows(rung, split)
    X = W[:, :-1].astype(np.int64)
    x0, z = parts(model, X)
    lp = finish(model, x0, z)                      # the unpatched model, float64
    assert np.abs(lp - model_logprobs(model, W)).max() < 1e-4
    claims = load_account(account).CLAIMS
    sites = ["direct"] + [f"head{h}" for h in range(model.cfg.n_heads)]
    named = [c["site"] for c in claims]
    assert sorted(named) == sorted(sites), f"claims must cover each of {sites} exactly once; got {named}"
    how = {}
    for c in claims:
        assert ("key" in c) != ("value" in c), f"claim {c['name']}: give exactly one of key, value"
        if "key" in c:
            how[c["site"]] = ("key", key_ids(c["key"](X)))
        else:
            v = np.asarray(c["value"](X), np.float64)
            want = x0.shape if c["site"] == "direct" else z.shape[:2] + z.shape[3:]
            assert v.shape == want, f"claim {c['name']}: value shape {v.shape}, site has {want}"
            how[c["site"]] = ("value", v)
    res = {"rung": rung, "arm": arm, "seed": seed, "split": split, "draws": DRAWS, "claims": {}}

    def faith(which: list[str]) -> dict:
        ks, kr = [], []
        for draw in range(DRAWS):
            rng = np.random.default_rng([seed, draw, len(which)])
            ks.append(kl_pos(lp, scrubbed(model, x0, z, {s: how[s] for s in which}, rng)).mean())
            kr.append(kl_pos(lp, scrubbed(model, x0, z, {s: None for s in which}, rng)).mean())
        ks, kr = float(np.mean(ks)), float(np.mean(kr))
        return {"kl_scrubbed": ks, "kl_resampled": kr, "faithfulness": 1 - ks / kr if kr > 0 else 1.0}

    ok = True
    for c in claims:
        s = c["site"]
        r = faith([s])
        r["test"] = "swap" if how[s][0] == "key" else "replace"
        if how[s][0] == "key":
            _, has = donors(how[s][1], np.random.default_rng(0))
            r["no_donor_share"] = float(1 - has.mean())
            r["n_key_values"] = int(how[s][1].max() + 1)
        r["pass"] = bool(r["faithfulness"] >= MIN_FAITHFULNESS and r.get("no_donor_share", 0) <= MAX_NO_DONOR)
        ok &= r["pass"]
        res["claims"][c["name"]] = {"site": s, **r}
    res["joint"] = faith(sites)
    res["joint"]["pass"] = bool(res["joint"]["faithfulness"] >= MIN_FAITHFULNESS)
    res["P3"] = bool(ok and res["joint"]["pass"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--account", required=True)
    ap.add_argument("--split", default="dev_full", choices=["dev_full", "dev_fast"])
    a = ap.parse_args()
    print(json.dumps(run(a.rung, a.arm, a.seed, a.account, a.split), indent=1))


if __name__ == "__main__":
    main()
