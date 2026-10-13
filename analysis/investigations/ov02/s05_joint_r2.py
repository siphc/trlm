"""s05: why R^2(heads 0+2 together) > R^2(head 0), R^2(head 2) — in the exact setting of analysis/vote_fit.head_r2.

y_h = lp - lp_ablated (harness 'earlier' ablation of head h; h = 0, 2 and both), weighted-centred per position;
X = program parts' earlier votes (local0, local2, recall, phrase), centred as in head_r2. Decompose
y_h = yhat_h + r_h (pooled weighted least squares) and report: SS, R^2, the correlation of yhat_0 with yhat_2, of r_0 with
r_2, and of y_0 with y_2; and the joint R^2 predicted from the single-head pieces.
"""
import json
import os

import numpy as np

from common import OUT
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from harness.common import head_means, load_model
from program import phrase
from program.tests import measure as M

P = program()
W = train_windows(16, seed=EVAL_SEED)
lp = model_logprobs(W)
means = head_means(load_model("attn_s0"), W)
Y = {k: model_logprobs(W, ablate_heads=h, ablation="earlier", means=means) for k, h in (("0", [0]), ("2", [2]), ("02", [0, 2]))}
feats = []
for w in W:
    t = np.asarray(w[:256], dtype=np.int64)
    feats.append({"local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
                  "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, P.STATS)})
ks = list(feats[0])
for k in ks:
    mu = np.mean([f[k].mean(0) for f in feats], axis=0)
    for f in feats:
        f[k] = f[k] - mu


def centred(x, w):
    return x - (w * x).sum(1, keepdims=True)


# pooled normal equations for each target
G = np.zeros((len(ks), len(ks)))
b = {k: np.zeros(len(ks)) for k in Y}
for f, p in zip(feats, lp):
    w = np.exp(p)
    X = np.stack([centred(f[k], w) for k in ks], -1)
    G += np.einsum("nvk,nv,nvl->kl", X, w, X)
for f, p, i in zip(feats, lp, range(len(lp))):
    w = np.exp(p)
    X = np.stack([centred(f[k], w) for k in ks], -1)
    for key in Y:
        y = centred(p - Y[key][i], w)
        b[key] += np.einsum("nvk,nv,nv->k", X, w, y)
beta = {k: np.linalg.solve(G, b[k]) for k in Y}
# second pass: sums of squares and cross products of fitted parts and residuals
acc = {}
def add(name, val):
    acc[name] = acc.get(name, 0.0) + val
for f, p, i in zip(feats, lp, range(len(lp))):
    w = np.exp(p)
    X = np.stack([centred(f[k], w) for k in ks], -1)
    y = {k: centred(p - Y[k][i], w) for k in Y}
    yh = {k: X @ beta[k] for k in Y}
    r = {k: y[k] - yh[k] for k in Y}
    for k in Y:
        add(f"ss_{k}", (w * y[k] ** 2).sum()); add(f"ess_{k}", (w * yh[k] ** 2).sum()); add(f"rss_{k}", (w * r[k] ** 2).sum())
    add("y0y2", (w * y["0"] * y["2"]).sum()); add("h0h2", (w * yh["0"] * yh["2"]).sum()); add("r0r2", (w * r["0"] * r["2"]).sum())
    add("y02_vs_sum", (w * (y["02"] - y["0"] - y["2"]) ** 2).sum())
res = {f"R2_{k}": 1 - acc[f"rss_{k}"] / acc[f"ss_{k}"] for k in Y}
res["corr_y0_y2"] = acc["y0y2"] / np.sqrt(acc["ss_0"] * acc["ss_2"])
res["corr_fitted0_fitted2"] = acc["h0h2"] / np.sqrt(acc["ess_0"] * acc["ess_2"])
res["corr_resid0_resid2"] = acc["r0r2"] / np.sqrt(acc["rss_0"] * acc["rss_2"])
res["ss_0"], res["ss_2"], res["ss_02"] = acc["ss_0"], acc["ss_2"], acc["ss_02"]
res["nonadditivity_share"] = acc["y02_vs_sum"] / acc["ss_02"]
# prediction of the joint R^2 if y02 = y0 + y2: ESS(y0+y2) <= (ess0 + ess2 + 2 h0h2) (fitted parts add; the LS fit of the
# sum is the sum of the fits), SS = ss0 + ss2 + 2 y0y2
res["R2_02_predicted_from_parts"] = (acc["ess_0"] + acc["ess_2"] + 2 * acc["h0h2"]) / (acc["ss_0"] + acc["ss_2"] + 2 * acc["y0y2"])
res["R2_02_if_fitted_parts_uncorrelated"] = (acc["ess_0"] + acc["ess_2"]) / (acc["ss_0"] + acc["ss_2"] + 2 * acc["y0y2"])
res["R2_02_if_residuals_uncorrelated_too"] = (acc["ess_0"] + acc["ess_2"] + 2 * acc["h0h2"]) / (
    acc["ss_0"] + acc["ss_2"] + 2 * acc["h0h2"])
res["beta"] = {k: dict(zip(ks, np.round(v, 3).tolist())) for k, v in beta.items()}
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
json.dump(res, open(os.path.join(OUT, "s05.json"), "w"), indent=1)
