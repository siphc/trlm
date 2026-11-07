"""P4, behaviour tests (logs/ladder/design.md; decision 19).

Each claim in an account's CLAIMS may carry
    "tests": [{"statement": "...", "measure": f, "op": "<=" or ">=", "threshold": x}, ...]
(or each test as a tuple (statement, measure, op, threshold))
where f(probe) returns one number measured on the MODEL (not the account). probe gives, on dev_full:
    probe.X       (n, T) input symbols
    probe.attn    (n, H, T, T) attention weights per head (query rows, key columns)
    probe.x0      (n, T, d) the direct-path embedding
    probe.z       (n, T, H, d_head) each head's output before the output projection
    probe.logp    (n, T, V) the model's log-probs;  probe.positions (T, d) its position embedding
and these costs, each KL(model || changed model) in nats per symbol:
    probe.cost_resample(site)          the site ("direct" or "head<h>") resample-ablated (one donor draw, seed 0)
    probe.cost_replaced(x0=, z=)       the direct path and/or the heads' outputs replaced by the given arrays
    probe.cost_positions(P)            the model rerun with its position embedding replaced by P (T, d)
The test code is the same for every seed (one account structure); it may read the account's per-seed constants
(the account file of that seed is the one loaded).

Decision 20: "measure" is normally a string naming a measurement of this module (MEASURES), so the account carries
only the claim, not the measuring code. Forms: "<name> <args>" or "<name> <args> / <name> <args>" (a ratio,
infinite when the denominator is at most 0.01 nats, i.e. when the reference part does not matter). Arguments that
are not sites or numbers name a constant of the account (a list of symbol ids). A callable is still accepted and
counts as account code.

A test passes if it holds on all five trained seeds and fails on the rung's untrained model (initialised with seed 0,
measured with seed 0's account file), so a test that every model passes cannot count. P4 passes if every claim has at
least one test and every test passes.

    .venv/bin/python -m analysis.ladder.behaviour --rung L1 --arm ln --account "ladder/L1_ln/account_s{seed}.py"
"""
from __future__ import annotations

import argparse
import json
import math
import operator

import numpy as np
import torch

from analysis.ladder.common import SEEDS, kl_pos, load, load_account, windows
from analysis.ladder.scrub import finish, parts, scrubbed
from src.train.model import TinyLM

OPS = {"<=": operator.le, ">=": operator.ge}


class Probe:
    def __init__(self, model, W: np.ndarray):
        model = model.double().eval()
        self.X = W[:, :-1].astype(np.int64)
        self.x0, self.z = parts(model, self.X)
        self._model = model
        self._attn = None
        self._logp = None

    @property
    def attn(self) -> np.ndarray:
        if self._attn is None:
            self._attn = attention(self._model, self.X)
        return self._attn

    @property
    def logp(self) -> np.ndarray:
        if self._logp is None:
            self._logp = finish(self._model, self.x0, self.z)
        return self._logp

    def cost_resample(self, site: str) -> float:
        return float(kl_pos(self.logp, scrubbed(self._model, self.x0, self.z, {site: None},
                                                np.random.default_rng(0))).mean())

    def cost_replaced(self, x0=None, z=None) -> float:
        lp = finish(self._model, self.x0 if x0 is None else x0, self.z if z is None else z)
        return float(kl_pos(self.logp, lp).mean())

    @torch.no_grad()
    def cost_positions(self, P: np.ndarray) -> float:
        m = self._model
        old = m.wpe.weight.clone()
        m.wpe.weight.copy_(torch.as_tensor(P, dtype=old.dtype))
        try:
            lp = torch.log_softmax(m(torch.as_tensor(self.X)), -1).numpy()
        finally:
            m.wpe.weight.copy_(old)
        return float(kl_pos(self.logp, lp).mean())

    @property
    def positions(self) -> np.ndarray:
        return self._model.wpe.weight.detach().numpy().copy()


@torch.no_grad()
def attention(model, X: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(X, dtype=torch.int64)
    n, T = x.shape
    blk = model.blocks[0]
    d, H, dh = model.cfg.d_model, model.cfg.n_heads, model.cfg.d_head
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, _ = blk.attn.qkv(blk.ln1(x0)).split(d, dim=-1)
    q, k = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k))
    s = q @ k.transpose(-1, -2) / math.sqrt(dh)
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return s.softmax(-1).numpy()


def _mean_for_current_symbol(X: np.ndarray, a: np.ndarray, V: int) -> np.ndarray:
    flat = a.reshape(X.size, -1)
    total = np.zeros((V, flat.shape[1]))
    np.add.at(total, X.ravel(), flat)
    return (total / np.maximum(np.bincount(X.ravel(), minlength=V), 1)[:, None])[X].reshape(a.shape)


def m_resample(p, site):
    """KL cost of resample-ablating the site ("direct" or "head<h>")."""
    return p.cost_resample(site)


def m_positions_mean(p):
    """KL cost of replacing every position vector by their mean."""
    P = p.positions
    return p.cost_positions(0 * P + P.mean(0))


def m_context(p, site):
    """KL cost of replacing a head's output by its mean for the current symbol (what the head adds through context)."""
    h = int(site[4:])
    z = p.z.copy()
    z[:, :, h] = _mean_for_current_symbol(p.X, p.z[:, :, h], p.logp.shape[-1])
    return p.cost_replaced(z=z)


def m_content_share(p, site):
    """Share of the variance of a head's log attention weights that the (query symbol, key symbol) pair explains.
    Queries t >= 32 (every 4th), keys 16 .. t, log weights centred per query."""
    h = int(site[4:])
    q, k, logw = [], [], []
    for t in range(32, p.X.shape[1], 4):
        row = np.log(p.attn[:, h, t, 16:t + 1] + 1e-30)
        logw.append((row - row.mean(1, keepdims=True)).ravel())
        q.append(np.repeat(p.X[:, t], t - 15))
        k.append(p.X[:, 16:t + 1].ravel())
    q, k, logw = (np.concatenate(a) for a in (q, k, logw))
    pair = q * p.logp.shape[-1] + k
    mean = np.bincount(pair, logw) / np.maximum(np.bincount(pair), 1)
    return float(1 - ((logw - mean[pair]) ** 2).mean() / (logw ** 2).mean())


def m_attention_ratio(p, site, symbols):
    """A head's attention weight on the given symbols divided by their share of the keys (queries t >= 16)."""
    h = int(site[4:])
    b = np.isin(p.X, symbols)
    mass = (p.attn[:, h, 16:] * b[:, None, :]).sum(-1).mean()
    share = np.mean([b[:, :t + 1].mean() for t in range(16, p.X.shape[1])])
    return float(mass / share)


MEASURES = {"resample": m_resample, "positions_mean": m_positions_mean, "context": m_context,
            "content_share": m_content_share, "attention_ratio": m_attention_ratio}
MIN_REFERENCE = 0.01


def measure(spec, probe, account) -> float:
    if callable(spec):
        return float(spec(probe))
    if " / " in spec:
        a, b = (measure(x, probe, account) for x in spec.split(" / "))
        return a / b if b > MIN_REFERENCE else float("inf")
    name, *args = spec.split()
    vals = [a if a == "direct" or a.startswith("head") else getattr(account, a) for a in args]
    return float(MEASURES[name](probe, *vals))


def untrained(model):
    torch.manual_seed(0)
    return TinyLM(model.cfg)


def run(rung: str, arm: str, account_pattern: str) -> dict:
    W = windows(rung, "dev_full")
    res = {"rung": rung, "arm": arm, "account": account_pattern, "claims": {}}
    models = {f"s{s}": (load(rung, arm, s), account_pattern.format(seed=s)) for s in SEEDS}
    models["untrained"] = (untrained(models["s0"][0]), account_pattern.format(seed=0))
    measured = {}   # (claim name, test index) -> tag -> (statement, op, threshold, value)
    for tag, (model, path) in models.items():
        probe = Probe(model, W)
        acc = load_account(path)
        for c in acc.CLAIMS:
            for i, t in enumerate(c.get("tests", [])):
                if isinstance(t, tuple):   # (statement, measure, op, threshold)
                    t = dict(zip(("statement", "measure", "op", "threshold"), t))
                measured.setdefault((c["name"], i), {})[tag] = (t["statement"], t["op"], float(t["threshold"]),
                                                                measure(t["measure"], probe, acc))
    names = [c["name"] for c in load_account(account_pattern.format(seed=0)).CLAIMS]
    ok = True
    for name in names:
        rows = []
        for (cname, i), per in sorted(measured.items()):
            if cname != name:
                continue
            holds = {tag: bool(OPS[op](v, thr)) for tag, (_, op, thr, v) in per.items()}
            passed = all(holds.get(f"s{s}", False) for s in SEEDS) and not holds["untrained"]
            st, op, thr, _ = per["s0"]
            rows.append({"statement": st, "op": op, "threshold": thr,
                         "values": {tag: v[3] for tag, v in per.items()}, "holds": holds, "pass": passed})
            ok &= passed
        ok &= bool(rows)
        res["claims"][name] = {"tests": rows, "pass": bool(rows) and all(r["pass"] for r in rows)}
    res["P4"] = bool(ok)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--account", required=True, help="path pattern with {seed}")
    a = ap.parse_args()
    print(json.dumps(run(a.rung, a.arm, a.account), indent=1))


if __name__ == "__main__":
    main()
