"""Statistics for the RECALL saturation claim (program/tests/test_head1.py), on TRAIN windows.

Per position: size of model head 1's earlier-token effect (lp - lp with head 1's earlier part at its mean; centred
and probability-weighted standard deviation over the vocabulary), size of RECALL's earlier-token votes with and
without the saturation (same measure, weights = model probabilities). Spearman correlations across positions.

    .venv/bin/python -m analysis.program_gap.saturation_claim [--n 32]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program


def wsize(x, w):
    mu = (x * w).sum(-1, keepdims=True)
    return np.sqrt((w * (x - mu) ** 2).sum(-1))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    from harness.common import head_means, load_model
    from program import head1
    from program.readers import earlier_only

    P = program()
    s = P.STATS
    W = train_windows(args.n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    m, sat, raw = [], [], []
    for w, p, pa in zip(W, lp, lpa):
        t = np.asarray(w[:256], np.int64)
        a = head1.attention(t, s)
        rv = head1.recalled_votes(t, s)
        e_raw = earlier_only(a) @ rv
        e_sat = head1.earlier_votes(a, t, s)
        pw = np.exp(p)
        m.append(wsize(p - pa, pw)); sat.append(wsize(e_sat, pw)); raw.append(wsize(e_raw, pw))
    m, sat, raw = map(np.concatenate, (m, sat, raw))
    res = {"n_windows": args.n, "spearman(model effect, saturated vote)": spearman(m, sat),
           "spearman(model effect, raw vote)": spearman(m, raw)}
    q = np.quantile(raw, [0.4, 0.6, 0.9])
    top, mid = raw >= q[2], (raw >= q[0]) & (raw < q[1])
    res["top decile / middle fifth of raw vote size"] = {
        "model effect": float(m[top].mean() / m[mid].mean()), "raw vote": float(raw[top].mean() / raw[mid].mean()),
        "fixed-size vote": float(sat[top].mean() / sat[mid].mean())}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
