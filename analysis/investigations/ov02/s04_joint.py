"""s04: why are heads 0 and 2 explained better together than apart? Cross-head structure of the earlier-token effects.

Exact linear decomposition in logit space: at position i, head h's earlier-token part writes
c_h(i) = LNf_i( (sum_{j<i} a_ij v_j - mean_ctx_h) W_O^h ) W_E^T, with the final LN linearised at the actual scale of
position i (the same scale for every term, so the terms add up exactly to the change of the logits, apart from the
change of the LN scale itself). Also the self part s_h(i) = a_ii v_i (minus its mean) and the direct path.
Weighted (model probabilities) per-position centring; pooled covariances over positions.
"""
import json
import os

import numpy as np

from common import OUT, V, DH, ln, model_weights
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs

m = model_weights()
W = train_windows(16, seed=EVAL_SEED)


def parts(t):
    n = len(t)
    x0 = m.WE[t] + m.WP[:n]
    h1, _ = ln(x0, m.g1)
    mask = np.tril(np.ones((n, n), dtype=bool))
    earlier, selfp, outs = [], [], []
    for h in range(4):
        q, k, v = h1 @ m.WQ[h].T, h1 @ m.WK[h].T, h1 @ m.WV[h].T
        s = np.where(mask, q @ k.T / np.sqrt(DH), -np.inf)
        s -= s.max(1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(1, keepdims=True)
        z = a @ v
        zs = np.diagonal(a)[:, None] * v
        earlier.append((z - zs) @ m.WOh[h].T)
        selfp.append(zs @ m.WOh[h].T)
    resid = x0 + sum(earlier) + sum(selfp)
    mu = resid.mean(-1, keepdims=True)
    sd = np.sqrt(((resid - mu) ** 2).mean(-1, keepdims=True) + 1e-5)

    def to_logits(c):
        c = c - c.mean(-1, keepdims=True)
        return (c * m.gf / sd) @ m.WE.T
    return {"E": [to_logits(c) for c in earlier], "S": [to_logits(c) for c in selfp], "DP": to_logits(x0)}


lp = model_logprobs(W)
P = [parts(np.asarray(w[:256], dtype=np.int64)) for w in W]
# subtract the mean over all positions (the 'mean' replacement in the harness ablation, in logit space)
keys = [("E", h) for h in range(4)] + [("S", h) for h in range(4)]
mean = {k: np.mean([p[k[0]][k[1]].mean(0) for p in P], axis=0) for k in keys}


def vec(p, k):
    return p[k[0]][k[1]] - mean[k]


def cov_matrix(items):
    """items: list of functions p -> (n, V); pooled weighted covariance matrix."""
    k = len(items)
    C = np.zeros((k, k))
    for p, l in zip(P, lp):
        w = np.exp(l)
        X = [f(p) for f in items]
        X = [x - (w * x).sum(1, keepdims=True) for x in X]
        for i in range(k):
            for j in range(i, k):
                C[i, j] += (w * X[i] * X[j]).sum()
                C[j, i] = C[i, j]
    return C


names = ["E0", "E1", "E2", "E3", "S0", "S1", "S2", "S3"]
items = [lambda p, k=k: vec(p, k) for k in keys]
C = cov_matrix(items)
sd = np.sqrt(np.diag(C))
R = C / np.outer(sd, sd)
print("pooled weighted std (logit units) per part:", dict(zip(names, np.round(sd / np.sqrt(sum(l.shape[0] * 1.0 for l in lp)), 3))))
print("correlation matrix:")
print("      " + " ".join(f"{n:>6}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>6} " + " ".join(f"{R[i, j]:6.2f}" for j in range(len(names))))
var02 = C[0, 0] + C[2, 2] + 2 * C[0, 2]
print("var(E0+E2) / (var E0 + var E2) = %.3f" % (var02 / (C[0, 0] + C[2, 2])))
allE = C[:4, :4].sum()
print("var(E0+E1+E2+E3) / sum var = %.3f" % (allE / np.trace(C[:4, :4])))
res = {"names": names, "corr": R.tolist(), "std": (sd).tolist(), "var02_ratio": float(var02 / (C[0, 0] + C[2, 2])),
       "varE_all_ratio": float(allE / np.trace(C[:4, :4]))}
json.dump(res, open(os.path.join(OUT, "s04.json"), "w"), indent=1)
