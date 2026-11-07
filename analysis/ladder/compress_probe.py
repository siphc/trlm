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

    OMP_NUM_THREADS=4 .venv/bin/python -m analysis.ladder.compress_probe --rung L1 --arm ln --seed 0
Writes analysis/out/ladder/compress_probe_<rung>_<arm>_s<seed>.json.
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
    def __init__(self, model, groups=None, offset=False, positions2=False):
        super().__init__()
        W = {k: v.detach().double().clone() for k, v in model.state_dict().items()}
        self.H, self.ln = model.cfg.n_heads, model.cfg.layernorm
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
        q, k, v = (u @ r["blocks_0_attn_qkv_weight"].T).split(d, -1)
        H, dh = self.H, d // self.H
        q, k, v = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k, v))
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(n, T, d)
        out = norm(h + z @ r["blocks_0_attn_out_weight"].T, r["ln_f_weight"] if self.ln else None) @ E.T
        if self.offset is not None:
            out = out + self.offset
        return out


def fit(model, version, rung, steps=3000, lr=3e-3, bs=64, seed=0):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="L1")
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    torch.manual_seed(0)
    model = load(a.rung, a.arm, a.seed).double()
    Wn = windows(a.rung, "dev_full")
    lp = model_logprobs(model, Wn)
    floor = float(kl_pos(lp, floor_logprobs(a.rung, a.arm, a.seed, Wn)).mean())
    alpha = data_char.load_alphabet()
    freq = np.array([2_099_719, alpha["unk_count"]] + alpha["counts"], np.float64)
    E = model.wte.weight.detach().numpy()
    n_params = sum(p.numel() for p in model.parameters())
    res = {"rung": a.rung, "arm": a.arm, "seed": a.seed, "n_params": n_params, "kl_floor": floor, "versions": []}
    specs = [("model refit (sanity)", {}),
             ("positions 2", {"positions2": True})]
    for k in (4, 8, 16, 32):
        specs += [(f"groups {k}, no offset", {"groups": k}), (f"groups {k} + offset", {"groups": k, "offset": True}),
                  (f"groups {k} + offset, positions 2", {"groups": k, "offset": True, "positions2": True})]
    for name, kw in specs:
        if "groups" in kw:
            kw = {**kw, "groups": kmeans(E, np.sqrt(freq), kw["groups"])}
        v = fit(model, Version(model, **kw), a.rung, a.steps)
        with torch.no_grad():
            lpv = np.concatenate([torch.log_softmax(v(torch.as_tensor(Wn[i:i + 200, :-1].astype(np.int64))), -1)
                                  .numpy() for i in range(0, len(Wn), 200)])
        kl = float(kl_pos(lp, lpv).mean())
        row = {"version": name, "free_numbers": v.n_free(), "kl": kl, "reaches_floor": kl <= max(floor, 0.01)}
        res["versions"].append(row)
        print(f"{name:40s} free {row['free_numbers']:4d} (model {n_params})  KL {kl:.4f}  floor {floor:.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"compress_probe_{a.rung}_{a.arm}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
