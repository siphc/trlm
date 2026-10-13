"""Step 7: what limits the explanation of head 3: its attention rule or its votes?

y = model log-probs minus log-probs with head 3's earlier-token part mean-ablated. Pooled weighted R^2 of y (as in
analysis/vote_fit.head_r2: features minus their mean over positions, then centred per position under the model's
probabilities) on feature sets, overall and by query group (the card's 7 groups).

  exact_dla   head 3's exact direct logit effect from earlier tokens (sanity: near 1)
  exA_U       exact attention x the model's token-level vote table U (mean position; model-derived, diagnostic)
  ruleA_U     card's attention rule x U          -> how much the attention rule loses
  exA_pmi     exact attention x PMI_{d+1}        -> how much the corpus vote loses
  ruleA_pmi   rule attention x PMI_{d+1}
  exA_cond1   exact attention at d = 1 x [log P(v | s q) - log P1(v | q)] (trigram conditional)
  local0, local2, recall, phrase: the program's parts (earlier-token votes / phrase votes)
"""
import json
import sys

import numpy as np

from h3lib import H3Run, T, V, OUT, tok_strs
import h3rule
from analysis.fitlib import train_windows
from analysis.mech_eval import program
from program.tests import measure as M
from program import phrase

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
P = program()
s = P.STATS
W = train_windows(NW, seed=SEED)[:, :T]
R = H3Run()
mc = R.mean_ctx(W)
U = np.load(f"{OUT}/U_model.npy").astype(np.float64)
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
logf = np.asarray(s.log_follow, np.float64)
tk, tc = np.asarray(s.tri_keys), np.asarray(s.tri_counts)
G = h3rule.group_of()
dist = np.arange(T)[:, None] - np.arange(T)[None, :]


def pmi_votes(A, t):
    out = np.zeros((len(t), V))
    for k in range(2, 9):
        m = (dist == k - 1) if k < 8 else (dist >= 7)
        out += (A * m) @ pmi[k][t]
    return out


def cond1(A, t):
    out = np.zeros((len(t), V))
    for i in range(1, len(t)):
        src, q = t[i - 1], t[i]
        key = src * V + q
        lo, hi = np.searchsorted(tk, key * V), np.searchsorted(tk, key * V + V)
        c = np.zeros(V)
        c[tk[lo:hi] - key * V] = tc[lo:hi]
        pc = (c + 20 * np.exp(logf[q])) / (c.sum() + 20)
        out[i] = A[i, i - 1] * (np.log(pc) - logf[q])
    return out


def features(r, t):
    A = r["a3"] * (dist >= 1)
    Ar = h3rule.attention(t) * (dist >= 1)
    u = R.source_votes(r)
    mcu = ((mc @ R.WO.T) - (mc @ R.WO.T).mean()) * R.net.gf @ R.net.WE.T
    return {"exact_dla": (A @ u - mcu[None]) / r["sdf"][:, None], "exA_U": A @ U[t], "ruleA_U": Ar @ U[t],
            "exA_pmi": pmi_votes(A, t), "ruleA_pmi": pmi_votes(Ar, t), "exA_cond1": cond1(A, t),
            "local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
            "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, s)}


runs = []
for w in W:
    r = R.run(w, mc)
    runs.append((np.asarray(w[:T], np.int64), r["lp"], r["lp"] - r["lpa"]))
names = None
mu = None
for t, lp, y in runs:     # pass 1: feature means over positions
    r = R.run(t, mc)
    f = features(r, t)
    names = list(f)
    m = np.stack([f[k].mean(0) for k in names])
    mu = m if mu is None else mu + m
mu /= len(runs)
groups = ["all"] + h3rule.GNAMES
K = len(names)
acc = {g: [np.zeros((K, K)), np.zeros(K), 0.0] for g in groups}
for t, lp, y in runs:
    r = R.run(t, mc)
    f = features(r, t)
    X = np.stack([f[k] for k in names], -1) - mu.T[None]
    w = np.exp(lp)
    yc = y - (w * y).sum(1, keepdims=True)
    X = X - (w[..., None] * X).sum(1, keepdims=True)
    qg = G[t]
    for g in groups:
        m = np.ones(len(t), bool) if g == "all" else qg == h3rule.GNAMES.index(g)
        if not m.any():
            continue
        acc[g][0] += np.einsum("nvk,nv,nvl->kl", X[m], w[m], X[m])
        acc[g][1] += np.einsum("nvk,nv,nv->k", X[m], w[m], yc[m])
        acc[g][2] += (w[m] * yc[m] ** 2).sum()


def r2(g, keys):
    idx = [names.index(k) for k in keys]
    xtx, xty, sst = acc[g][0][np.ix_(idx, idx)], acc[g][1][idx], acc[g][2]
    b = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3)


sets = [["exact_dla"], ["exA_U"], ["ruleA_U"], ["exA_pmi"], ["ruleA_pmi"], ["exA_cond1"], ["phrase"],
        ["local0", "local2", "recall", "phrase"], ["local0", "local2", "recall", "phrase", "exA_pmi"],
        ["local0", "local2", "recall", "phrase", "ruleA_pmi"], ["local0", "local2", "recall", "phrase", "exA_cond1"]]
res = {}
for g in groups:
    res[g] = {"sst_share": round(acc[g][2] / acc["all"][2], 3), **{"+".join(k): r2(g, k) for k in sets}}
    print(g, res[g], flush=True)
json.dump(res, open(f"{OUT}/oracle_r2_s{SEED}.json", "w"), indent=1)
