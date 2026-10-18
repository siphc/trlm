"""What is the model's current-token table made of? (analysis only)

B(s, v): the model's current-token logit contribution (direct path + every head's current-token part, final LN at
the actual per-position scale), centred over v, averaged over the positions of TRAIN windows whose current token
is s (tokens with >= MIN positions). Regress B(s, .) on corpus features, pooled over s, weighted by the model's
probability mass there (what KL cares about) and, separately, uniformly over v.

    .venv/bin/python -m analysis.program_gap.table_probe [--n 256]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.program_gap.bottleneck import OUT
from analysis.program_gap.budget import model_parts
from analysis.mech_eval import program

V = 2000
MIN = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    from harness.common import load_model

    torch.set_num_threads(4)
    model = load_model("attn_s0")
    P = program()
    s = P.STATS
    W = train_windows(args.n, seed=7)
    acc, cnt, pm = np.zeros((V, V)), np.zeros(V), np.zeros((V, V))
    for w in W:
        t = np.asarray(w[:256], np.int64)
        mp, logits = model_parts(model, t)
        S = mp["direct"] + sum(mp[f"self{h}"] for h in range(4))
        np.add.at(acc, t, S - S.mean(-1, keepdims=True))
        np.add.at(cnt, t, 1)
        z = logits - logits.max(-1, keepdims=True)
        np.add.at(pm, t, np.exp(z) / np.exp(z).sum(-1, keepdims=True))
    keep = np.nonzero(cnt >= MIN)[0]
    B = acc[keep] / cnt[keep, None]
    wmass = pm[keep] / cnt[keep, None]                  # model's mean next-token distribution after s
    np.savez_compressed(os.path.join(OUT, "model_table.npz"), keep=keep, B=B, cnt=cnt[keep], wmass=wmass)
    lp1 = np.asarray(s.log_follow, np.float64)[keep]
    lu = np.broadcast_to(np.log(s.p_uni), B.shape)
    pmi2 = np.asarray(s.pmi[2], np.float64)[keep]
    alike = np.asarray(s.slot_affinity, np.float64)[keep]
    copy = np.zeros_like(B)
    copy[np.arange(len(keep)), keep] = 1.0
    nb = np.asarray(s.bigram_counts, np.float64)
    lcont = np.broadcast_to(np.log1p((nb > 0).sum(0)), B.shape)
    # successor-overlap similarity between s and v (do s and v predict the same next tokens?)
    r = np.sqrt(np.exp(np.asarray(s.log_follow, np.float64)))
    succ_sim = (r @ r.T)[keep]
    # predecessor-side: does v usually come right before s? (reverse bigram)
    rev = np.log((nb.T[keep] + 1) / (nb.sum(0)[keep, None] + V))
    piece = np.broadcast_to(s.piece, B.shape)
    feats = {"log_P1": lp1, "log_uni": lu, "pmi2": pmi2, "alike": alike, "copy": copy, "log_ncont": lcont,
             "succ_sim": succ_sim, "rev_bigram": rev, "piece": piece}
    for k in list(feats):
        feats[k] = feats[k] - feats[k].mean(1, keepdims=True)
    y = B
    res = {"n_windows": args.n, "n_tokens": int(len(keep))}
    for wname, Wt in (("model_mass", wmass * cnt[keep, None]), ("uniform", np.broadcast_to(cnt[keep, None], B.shape))):
        Wt = Wt / Wt.sum()
        yc = y - (Wt * y).sum(1, keepdims=True) / Wt.sum(1, keepdims=True)
        sst = (Wt * yc ** 2).sum()

        def r2(keys):
            X = np.stack([feats[k] - (Wt * feats[k]).sum(1, keepdims=True) / Wt.sum(1, keepdims=True) for k in keys], -1)
            xtx = np.einsum("svk,sv,svl->kl", X, Wt, X)
            xty = np.einsum("svk,sv,sv->k", X, Wt, yc)
            b = np.linalg.solve(xtx + 1e-9 * np.eye(len(keys)), xty)
            return float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), [round(float(x), 3) for x in b]

        out = {}
        for keys in (["log_P1"], ["log_P1", "log_uni"], ["log_P1", "log_uni", "copy"],
                     ["log_P1", "log_uni", "copy", "alike"], ["log_P1", "log_uni", "copy", "alike", "succ_sim"],
                     ["log_P1", "log_uni", "copy", "alike", "succ_sim", "pmi2", "rev_bigram", "log_ncont", "piece"]):
            out["+".join(keys)] = r2(keys)
        res[wname] = out
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "table_probe.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
