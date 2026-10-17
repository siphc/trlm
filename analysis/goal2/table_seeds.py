"""Is the model's current-token table shared across seeds? (goal 2, analysis only)

For attn_s0..s4: B_k(s, .) = the model's current-token logit contribution (direct + every head's current-token
part, final LN at the actual scale), centred, averaged over the positions of the same TRAIN windows whose current
token is s (>= MIN positions). R^2 of B_0 explained by (a) corpus features (log P1, log P), (b) the mean table of
seeds 1-4, (c) both; weighted by attn_s0's mean next-token mass (as in table_probe.py).

    .venv/bin/python -m analysis.goal2.table_seeds [--n 128]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT
from analysis.goal2.budget import model_parts
from analysis.mech_eval import program

V = 2000
MIN = 40


def tables(model, W):
    acc, cnt, pm = np.zeros((V, V)), np.zeros(V), np.zeros((V, V))
    for w in W:
        t = np.asarray(w[:256], np.int64)
        mp, logits = model_parts(model, t)
        S = mp["direct"] + sum(mp[f"self{h}"] for h in range(4))
        np.add.at(acc, t, S - S.mean(-1, keepdims=True))
        np.add.at(cnt, t, 1)
        z = logits - logits.max(-1, keepdims=True)
        np.add.at(pm, t, np.exp(z) / np.exp(z).sum(-1, keepdims=True))
    return acc, cnt, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    from harness.common import load_model

    torch.set_num_threads(4)
    P = program()
    s = P.STATS
    W = train_windows(args.n, seed=7)
    T = {}
    for k in range(5):
        acc, cnt, pm = tables(load_model(f"attn_s{k}"), W)
        T[k] = (acc, pm)
    keep = np.nonzero(cnt >= MIN)[0]
    B = {k: T[k][0][keep] / cnt[keep, None] for k in T}
    wm = T[0][1][keep] / cnt[keep, None]
    np.savez_compressed(os.path.join(OUT, "tables_seeds.npz"), keep=keep, cnt=cnt[keep], wm=wm,
                        **{f"B{k}": B[k].astype(np.float32) for k in B},
                        **{f"pm{k}": (T[k][1][keep] / cnt[keep, None]).astype(np.float32) for k in T})
    Wt = wm * cnt[keep, None]
    Wt = Wt / Wt.sum()

    def cen(x):
        return x - (Wt * x).sum(1, keepdims=True) / Wt.sum(1, keepdims=True)

    lp1 = np.asarray(s.log_follow, np.float64)[keep]
    lu = np.broadcast_to(np.log(s.p_uni), lp1.shape)
    others = np.mean([B[k] for k in range(1, 5)], 0)
    feats = {"log_P1": lp1, "log_uni": lu, "seeds1-4 table": others, "seed1 table": B[1]}
    res = {"n_windows": args.n, "n_tokens": int(len(keep))}
    for target in range(5):
        y = cen(B[target])
        sst = (Wt * y * y).sum()

        def r2(keys, fe):
            X = np.stack([cen(fe[k]) for k in keys], -1)
            xtx = np.einsum("svk,sv,svl->kl", X, Wt, X)
            xty = np.einsum("svk,sv,sv->k", X, Wt, y)
            b = np.linalg.solve(xtx, xty)
            return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3)

        fe = dict(feats)
        fe["mean of the other seeds"] = np.mean([B[k] for k in range(5) if k != target], 0)
        res[f"attn_s{target}"] = {"corpus (log P1, log P)": r2(["log_P1", "log_uni"], fe),
                                  "mean of the other 4 seeds": r2(["mean of the other seeds"], fe),
                                  "corpus + other seeds": r2(["log_P1", "log_uni", "mean of the other seeds"], fe)}
        print(target, res[f"attn_s{target}"], flush=True)
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "table_seeds.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
