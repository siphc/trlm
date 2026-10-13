"""Mean model predictions on TRAIN windows: the KL-optimal context-free predictors.

For KL(model || q) with q allowed to depend only on a feature f of the context, the optimum is
q(.|f) = E[p_model(.|context) | f]. This script accumulates those means for
  f = nothing          -> mean_unigram   (V,)
  f = current token    -> mean_bigram    (V, V)   row t = mean model distribution after token t
  f = position         -> mean_by_pos    (256, V)
and the KL each one achieves on the same windows.

Usage: python -m analysis.mean_dists --model attn_s0 [--n 4000]
Writes analysis/out/<model>/mean_dists.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from analysis.fitlib import REPO, Target, model_name, train_windows

V = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_name", default="mean_dists.npz")
    args = ap.parse_args()
    torch.set_num_threads(4)
    tgt = Target(args.model, args.device)
    W = torch.as_tensor(train_windows(args.n))
    S_uni = torch.zeros(V, dtype=torch.float64)
    S_big = torch.zeros(V, V, dtype=torch.float64)
    S_pos = torch.zeros(256, V, dtype=torch.float64)
    cnt = torch.zeros(V, dtype=torch.float64)
    H = 0.0
    for i in range(0, len(W), 50):
        x = W[i : i + 50, :256]
        lp = tgt.logprobs(x).cpu().double()
        p = lp.exp()
        H += float(-(p * lp).sum())
        S_uni += p.sum((0, 1))
        S_pos += p.sum(0)
        S_big.index_add_(0, x.reshape(-1), p.reshape(-1, V))
        cnt += torch.bincount(x.reshape(-1), minlength=V).double()
    N = len(W) * 256
    uni = S_uni / N
    big = S_big / cnt.clamp(min=1)[:, None]
    big[cnt == 0] = uni
    pos = S_pos / len(W)
    # KL(model || mean predictor) = E[sum p log p] - E[sum p log q]
    eps = 1e-12
    cross_uni = float(-(S_uni * (uni + eps).log()).sum()) / N
    cross_big = float(-(S_big * (big + eps).log()).sum()) / N
    cross_pos = float(-(S_pos * (pos + eps).log()).sum()) / N
    ent = H / N
    out = os.path.join(REPO, "analysis", "out", model_name(args.model))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, args.out_name), uni=uni.numpy(), big=big.numpy(), pos=pos.numpy(),
             count=cnt.numpy())
    print(f"model entropy {ent:.4f} nats/token over {N:,} train tokens")
    print(f"KL(model || mean unigram)            {cross_uni - ent:.4f}")
    print(f"KL(model || mean given position)     {cross_pos - ent:.4f}")
    print(f"KL(model || mean given current token) {cross_big - ent:.4f}")


if __name__ == "__main__":
    main()
