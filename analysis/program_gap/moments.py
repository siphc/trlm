"""Does the model satisfy the training equilibrium ("moment matching") better than the program? (analysis only)

A model trained by cross-entropy, at a stationary point, makes its average prediction over the positions where a
feature is active equal to the empirical next-token distribution there (projected onto what it can express). For
the current-token feature: for each token s, mean over positions with t_i = s of p_i(.) ~ the empirical
distribution of the next token after s. Same with attention weights for a head's earlier-token feature.

Measured on TRAIN windows, for the model and the program, per token s with n_s >= MIN positions:
  D(s) = KL(empirical next-token distribution after s || mean prediction after s), averaged weighted by n_s;
  also a shuffled reference: the same statistic for the model against a random half of the positions (noise floor).

    .venv/bin/python -m analysis.program_gap.moments [--n 256]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.program_gap.bottleneck import OUT
from analysis.mech_eval import model_logprobs, program

V = 2000
MIN = 30


def stat(emp, pred, n, eps=1e-9):
    ok = n >= MIN
    e = emp[ok] / n[ok, None]
    p = pred[ok] / n[ok, None]
    d = np.where(e > 0, e * (np.log(e + eps) - np.log(p + eps)), 0.0).sum(1)
    return float((d * n[ok]).sum() / n[ok].sum()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    P = program()
    W = train_windows(args.n, seed=7)
    emp = np.zeros((V, V))
    pm = np.zeros((V, V))
    pq = np.zeros((V, V))
    n = np.zeros(V)
    for c in range(0, args.n, 16):
        lp = model_logprobs(W[c:c + 16])
        for w, l in zip(W[c:c + 16], lp):
            t = np.asarray(w[:257], np.int64)
            cur, nxt = t[:256], t[1:257]
            lq = P.predict(t[:256])
            np.add.at(emp, (cur[:255], nxt[:255]), 1.0)
            np.add.at(pm, cur[:255], np.exp(l[:255]))
            np.add.at(pq, cur[:255], np.exp(lq[:255]))
            np.add.at(n, cur[:255], 1.0)
    res = {"n_windows": args.n, "min_positions": MIN}
    res["model"], res["n_tokens_used"] = stat(emp, pm, n)
    res["program"], _ = stat(emp, pq, n)
    # model vs program (how far their averages are from each other)
    res["model_vs_program_means"], _ = stat(pm, pq, n)
    # the corpus bigram from the whole TRAIN stream as the 'prediction' (sampling noise reference)
    big = np.asarray(P.STATS.bigram_counts, np.float64)
    pc = big / np.maximum(big.sum(1, keepdims=True), 1)
    res["corpus_bigram_as_prediction"], _ = stat(emp, pc * n[:, None], n)
    # whole-stream corpus bigram as the reference (much less noisy than the sample)
    res["model_vs_corpus_bigram"], _ = stat(pc * n[:, None], pm, n)
    res["program_vs_corpus_bigram"], _ = stat(pc * n[:, None], pq, n)
    res["model_vs_corpus_bigram_reverse"], _ = stat(pm, pc * n[:, None], n)
    res["program_vs_corpus_bigram_reverse"], _ = stat(pq, pc * n[:, None], n)
    np.savez_compressed(os.path.join(OUT, "moments_means.npz"), pm=pm, pq=pq, n=n)
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "moments.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
