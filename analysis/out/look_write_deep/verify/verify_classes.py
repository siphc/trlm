"""Recompute the head classes (LOCAL, PREV, INDUCTION, earlier mass, confirmatory) with an independent float64
forward pass and compare with classes.json.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_classes.py attn2L_s1 ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bgu import Net, REPO, T  # noqa: E402

sys.path.insert(0, REPO)


def patterns(net, x):
    with torch.no_grad():
        sd, L, H, D, DH = net.sd, net.L, net.H, net.D, net.DH
        out = []
        h = sd["wte.weight"][x] + sd["wpe.weight"][:x.shape[1]][None]
        from verify_bgu import ln
        import math
        TT = x.shape[1]
        causal = torch.ones(TT, TT, dtype=torch.bool).tril()
        B = x.shape[0]
        for l in range(L):
            xin = ln(h, sd[f"blocks.{l}.ln1.weight"])
            W = sd[f"blocks.{l}.attn.qkv.weight"]
            q = (xin @ W[:D].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            k = (xin @ W[D:2 * D].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            v = (xin @ W[2 * D:].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            s = torch.where(causal, q @ k.transpose(-1, -2) / math.sqrt(DH), torch.tensor(-math.inf))
            a = torch.softmax(s, -1)
            h = h + (a @ v).permute(0, 2, 1, 3).reshape(B, TT, D) @ sd[f"blocks.{l}.attn.out.weight"].T
            out.append(a.numpy())
    return out


def main(models):
    from src.train.data import load_stream

    s = load_stream("train")
    n_all = (len(s) - 1) // T
    fit_idx = np.sort(np.random.default_rng(2024).choice(n_all, size=32, replace=False))
    x = torch.as_tensor(np.stack([s[i * T:i * T + T] for i in fit_idx]).astype(np.int64))
    seq = np.random.default_rng(0).integers(1, 2000, size=(32, 128))
    xi = torch.as_tensor(np.concatenate([seq, seq], 1))
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    lower = dist > 0
    for m in models:
        net = Net(f"{REPO}/models/deep/{m}/final.pt")
        P, PI = patterns(net, x), patterns(net, xi)
        stored = {(c["layer"], c["head"]): c for c in json.load(open(f"{REPO}/analysis/out/look_write_deep/{m}/classes.json"))["heads"]}
        worst = {}
        for l in range(net.L):
            for h in range(net.H):
                a = P[l][:, h] * lower
                mass = a.sum(-1)[:, 1:]
                md = (a * dist).sum(-1)[:, 1:].sum() / mass.sum()
                em = mass.mean()
                prev = P[l][:, h, np.arange(1, T), np.arange(T - 1)].mean()
                ii = np.arange(129, 256)
                ind = PI[l][:, h, ii, ii - 127].mean()
                conf = md <= 10 and prev < 0.5 and ind < 0.3 and em >= 0.15
                c = stored[(l, h)]
                for k_, v_ in (("mean_distance", md), ("earlier_mass", em), ("prev_attention", prev),
                               ("induction_score", ind)):
                    worst[k_] = max(worst.get(k_, 0), abs(v_ - c[k_]))
                if conf != c["confirmatory"]:
                    print("  CLASS DIFF", m, l, h, conf, c["confirmatory"])
        print(m, "max |own - stored|:", {k: f"{v:.1e}" for k, v in worst.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
