"""Exploratory (decision 18 follow-up): can a description much shorter than the weights reach the seed floor at L1?

Structured versions of the L1 computation are fitted to the model's outputs (KL(model || version), on train windows;
scored on dev_full) and their free numbers counted. Each version keeps the model's computation (embedding +
position, pre-LN attention head, final LN, tied unembedding) and simplifies one part:
- "groups k": every symbol's embedding is one of k group vectors (groups from frequency-weighted k-means on the
  model's embedding), plus a per-symbol output offset added to the logits (79 numbers); positions as the model.
- "groups k, no offset": the same without the offset.
- "positions 2": the position embedding is one vector at position 0 and one shared by every later position.
- combinations.
Not an account (no claims, no stated meaning); it measures how many numbers the behaviour at the floor needs.

- "rank r": each head's QK and OV truncated to rank r (by SVD, then fitted).
The first run (fixed list of versions, L1 seed 0) is in logs/ladder/compress_probe.md; main() now runs a greedy
search for the fewest free numbers at or below the floor (see main's docstring), scored on dev_full[:1000].

    OMP_NUM_THREADS=4 .venv/bin/python -m analysis.ladder.compress_probe --rung L1 --arm ln --seed 0
Writes analysis/out/ladder/compress_search_<rung>_<arm>_s<seed>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from analysis.ladder.common import REPO, kl_pos, load, model_logprobs, windows
from analysis.ladder.eval import floor_logprobs
from src.train import data_char
from src.train.data import BatchSampler

OUT = os.path.join(REPO, "analysis", "out", "ladder")


def kmeans(E: np.ndarray, w: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = E[rng.choice(len(E), k, replace=False, p=w / w.sum())]
    for _ in range(200):
        a = ((E[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([np.average(E[a == j], 0, w[a == j]) if (a == j).any() else C[j] for j in range(k)])
    return a


class Version(torch.nn.Module):
    def __init__(self, model, groups=None, offset=False, positions2=False, rank=None):
        super().__init__()
        W = {k: v.detach().double().clone() for k, v in model.state_dict().items()}
        self.H, self.ln = model.cfg.n_heads, model.cfg.layernorm
        self.dh = model.cfg.d_head
        self.rank = rank
        if rank is not None:
            # per head: QK = Wq^T Wk and OV = Wo Wv truncated to rank r by SVD (in the normalised-input basis)
            d, H, dh = model.cfg.d_model, self.H, self.dh
            Wq, Wk, Wv = W.pop("blocks.0.attn.qkv.weight").split(d, 0)
            Wo = W.pop("blocks.0.attn.out.weight")
            q_, k_, v_, o_ = [], [], [], []
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                U, S, Vt = torch.linalg.svd(Wq[sl].T @ Wk[sl])
                q_.append((U[:, :rank] * S[:rank].sqrt()).T)
                k_.append((Vt[:rank].T * S[:rank].sqrt()).T)
                U, S, Vt = torch.linalg.svd(Wo[:, sl] @ Wv[sl])
                o_.append(U[:, :rank] * S[:rank].sqrt())
                v_.append(Vt[:rank] * S[:rank].sqrt()[:, None])
            self.Wq = torch.nn.Parameter(torch.stack(q_))      # (H, r, d)
            self.Wk = torch.nn.Parameter(torch.stack(k_))
            self.Wv = torch.nn.Parameter(torch.stack(v_))
            self.Wo = torch.nn.Parameter(torch.stack(o_))      # (H, d, r)
        self.groups = None
        if groups is not None:
            self.register_buffer("assign", torch.as_tensor(groups))
            k = int(groups.max() + 1)
            C = torch.stack([W["wte.weight"][self.assign == j].mean(0) for j in range(k)])
            self.C = torch.nn.Parameter(C)
            self.groups = k
        else:
            self.E = torch.nn.Parameter(W["wte.weight"])
        self.offset = torch.nn.Parameter(torch.zeros(model.cfg.vocab_size, dtype=torch.float64)) if offset else None
        if positions2:
            P = W["wpe.weight"]
            self.p0 = torch.nn.Parameter(P[0].clone())
            self.prest = torch.nn.Parameter(P[1:].mean(0))
            self.P = None
        else:
            self.P = torch.nn.Parameter(W["wpe.weight"])
        self.rest = torch.nn.ParameterDict({k.replace(".", "_"): torch.nn.Parameter(v) for k, v in W.items()
                                            if k not in ("wte.weight", "wpe.weight")})

    def n_free(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def emb(self):
        return self.C[self.assign] if self.groups else self.E

    def forward(self, x):
        n, T = x.shape
        E = self.emb()
        P = self.P[:T] if self.P is not None else torch.cat([self.p0[None], self.prest[None].expand(T - 1, -1)])
        h = E[x] + P[None]
        d = h.shape[-1]
        r = self.rest

        def norm(v, g):
            return F.layer_norm(v, (d,), g) if self.ln else v

        u = norm(h, r["blocks_0_ln1_weight"] if self.ln else None)
        H, dh = self.H, d // self.H
        if self.rank is None:
            q, k, v = (u @ r["blocks_0_attn_qkv_weight"].T).split(d, -1)
            q, k, v = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k, v))
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(n, T, d)
            write = z @ r["blocks_0_attn_out_weight"].T
        else:
            q = torch.einsum("ntd,hrd->nhtr", u, self.Wq)
            k = torch.einsum("ntd,hrd->nhtr", u, self.Wk)
            v = torch.einsum("ntd,hrd->nhtr", u, self.Wv)
            sc = q @ k.transpose(-1, -2) / np.sqrt(dh)
            sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
            z = sc.softmax(-1) @ v                                 # (n, H, T, r)
            write = torch.einsum("nhtr,hdr->ntd", z, self.Wo)
        out = norm(h + write, r["ln_f_weight"] if self.ln else None) @ E.T
        if self.offset is not None:
            out = out + self.offset
        return out


def fit(model, version, rung, steps=3000, lr=3e-3, bs=64, seed=0):
    """Fit `version` to the model's outputs: KL(model || version) on train windows, Adam, cosine to 10%."""
    ctx = model.cfg.context_length
    sampler = BatchSampler(data_char.load_stream("train"), 1000 + seed, bs, ctx)
    opt = torch.optim.Adam(version.parameters(), lr=lr)
    model = model.double()
    for step in range(steps):
        x = torch.as_tensor(sampler.batch(step))[:, :-1]
        with torch.no_grad():
            lp_m = torch.log_softmax(model(x), -1)
        lp_v = torch.log_softmax(version(x), -1)
        loss = (lp_m.exp() * (lp_m - lp_v)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
    return version


def evaluate_version(model, kw, Wn, lp, E, freq, steps):
    kw = dict(kw)
    if kw.get("groups"):
        kw["groups"] = kmeans(E, np.sqrt(freq), kw["groups"])
    else:
        kw.pop("groups", None)
    v = fit(model, Version(model, **kw), None, steps)
    with torch.no_grad():
        lpv = np.concatenate([torch.log_softmax(v(torch.as_tensor(Wn[i:i + 200, :-1].astype(np.int64))), -1).numpy()
                              for i in range(0, len(Wn), 200)])
    return v.n_free(), float(kl_pos(lp, lpv).mean())


def main() -> None:
    """Greedy search for the fewest free numbers at or below the floor: positions 2 first, then the attention rank
    (halving from d_head), then the number of symbol groups (from 64 down, with the per-symbol offset); each step is
    kept only if the fitted version stays at or below max(floor, 0.01)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="L1")
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    torch.manual_seed(0)
    model = load(a.rung, a.arm, a.seed).double()
    Wn = windows(a.rung, "dev_full")[:1000]
    lp = model_logprobs(model, Wn)
    floor = float(kl_pos(lp, floor_logprobs(a.rung, a.arm, a.seed, Wn)).mean())
    bar = max(floor, 0.01)
    alpha = data_char.load_alphabet()
    freq = np.array([2_099_719, alpha["unk_count"]] + alpha["counts"], np.float64)
    E = model.wte.weight.detach().numpy()
    n_params = sum(p.numel() for p in model.parameters())
    res = {"rung": a.rung, "arm": a.arm, "seed": a.seed, "n_params": n_params, "kl_floor": floor,
           "windows": "dev_full[:1000]", "steps": a.steps, "tried": []}

    def tryit(kw):
        n, kl = evaluate_version(model, kw, Wn, lp, E, freq, a.steps)
        row = {"version": {k: v for k, v in kw.items()}, "free_numbers": n, "kl": kl, "ok": kl <= bar}
        res["tried"].append(row)
        print(f"{json.dumps(row['version']):60s} free {n:6d} (model {n_params})  KL {kl:.4f}  bar {bar:.4f}",
              flush=True)
        return row["ok"]

    best = {}
    if tryit({"positions2": True}):
        best["positions2"] = True
    r = model.cfg.d_head
    while r > 1 and tryit({**best, "rank": r // 2}):
        r //= 2
        best["rank"] = r
    for k in (64, 48, 32, 24, 16, 12, 8):
        if k >= model.cfg.vocab_size:
            continue
        if tryit({**best, "groups": k, "offset": True}):
            best.update({"groups": k, "offset": True})
        else:
            break
    found = [t for t in res["tried"] if t["ok"]]
    res["best"] = min(found, key=lambda t: t["free_numbers"]) if found else None
    if res["best"]:
        res["ratio"] = n_params / res["best"]["free_numbers"]
        print(f"best: {res['best']['free_numbers']} free numbers, ratio {res['ratio']:.2f}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"compress_search_{a.rung}_{a.arm}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
