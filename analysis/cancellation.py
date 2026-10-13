"""Why are the program's head parts 5-10x weaker than the model's heads? (consistency check, mechanism v1)

Hypothesis: the model's heads write large opposing votes that cancel in the sum (e.g. head 0 vetoes repeating the
last few tokens while head 1 votes to repeat earlier words), and the program fitted only their net effect.

Test: regress the model's change under earlier-token mean ablation of head h (d_model, per position, weighted by
the model's probabilities, centred per position) on directions the program can express with the head's own
attention rule a_h:
  votes   the program part's earlier-token votes (skip-gram or recall)
  copy    sum_j<i a_h(i, j) [v = t_j]            (vote for the tokens the head reads)
  unigram sum_j<i a_h(i, j) log P(v)             (attention mass on earlier tokens times log P(v))
and report the weighted R^2 and the coefficients.

  python -m analysis.cancellation
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.fitlib import REPO, train_windows
from analysis.mech_eval import EVAL_SEED, HEAD_OF, model_logprobs, program
from program.tests import measure as M


def directions(P, name, t):
    from program import head0, head1, head2

    mod = {"head0": head0, "head1": head1, "head2": head2}[name]
    n = len(t)
    a = mod.attention(t, P.STATS) * (1.0 - np.eye(n))
    copy = np.zeros((n, M.V))
    np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), a.reshape(-1))
    uni = a.sum(1, keepdims=True) * P.STATS.log_uni[None, :].astype(np.float64)
    return {"votes": M.earlier_votes(P, name, t), "copy": copy, "unigram": uni}


def fit(dm_list, w_list, feats_list, keys):
    k = len(keys)
    xtx, xty, sst = np.zeros((k, k)), np.zeros(k), 0.0
    for dm, w, f in zip(dm_list, w_list, feats_list):
        X = np.stack([f[key] for key in keys], -1)                     # (n, V, k)
        X = X - (w[..., None] * X).sum(1, keepdims=True)
        y = dm - (w * dm).sum(1, keepdims=True)
        xtx += np.einsum("nvk,nv,nvl->kl", X, w, X)
        xty += np.einsum("nvk,nv,nv->k", X, w, y)
        sst += (w * y * y).sum()
    beta = np.linalg.solve(xtx, xty)
    sse = sst - 2 * beta @ xty + beta @ xtx @ beta
    return {"keys": keys, "beta": [float(b) for b in beta], "r2": float(1 - sse / sst)}


def main(n=32):
    from harness.common import head_means, load_model

    P = program()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    out = {}
    for name, h in HEAD_OF.items():
        lpa = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        feats = [directions(P, name, t) for t in toks]
        for key in feats[0]:                                           # centre each direction over positions
            mu = np.mean([f[key].mean(0) for f in feats], axis=0)
            for f in feats:
                f[key] = f[key] - mu
        dm = [p - pa for p, pa in zip(lp, lpa)]
        wts = [np.exp(p) for p in lp]
        out[name] = {"+".join(ks): fit(dm, wts, feats, ks)
                     for ks in [["votes"], ["copy"], ["votes", "copy"], ["votes", "copy", "unigram"]]}
        print(name, json.dumps(out[name]), flush=True)
    with open(os.path.join(REPO, "analysis", "out", "cancellation.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
