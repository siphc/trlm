"""Step 3: pooled R^2 of head 3's per-source votes u_j on corpus features (pairs with a_ij >= 0.3, j < i),
weighted by the model's next-token probabilities at the query. Also shows a few sources' votes next to PMI.

Features (each a function of s = t_j, v; some also of q = t_i, d = i - j):
  mean        the attention-weighted mean source vote (shared by all sources; model-derived, diagnostic only)
  pmi_match   PMI_{d+1}(s, v)
  pmi_k       PMI_k(s, v), k = 1..4
  cond        log P(v | s at -d, q) - log P1(v | q)  (d <= 2 only; 0 elsewhere)
  anti_q      log P1(v | q) (the current token's bigram)
  log_uni     log P(v)
  same_att    [ATTENTION class of v == that of s];  same_tgt  [TARGET class of v == TARGET class of s]
  par_tgt     log P(TARGET class of v at +2 | s)-style class PMI (class of v vs class of s at distance 2)
"""
import json
import sys

import numpy as np

from h3lib import H3Run, T, V, OUT, class_ids, tok_strs, wcorr
from analysis.fitlib import train_windows
from analysis.mech_eval import stats

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 12
W = train_windows(NW, seed=11)[:, :T]
R = H3Run()
mc = R.mean_ctx(W)
s = stats()
att, _ = class_ids("ATTENTION")
tgt, _ = class_ids("TARGET")
S = tok_strs()
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
logf = np.asarray(s.log_follow, np.float64)
tk, tc = np.asarray(s.tri_keys), np.asarray(s.tri_counts)
qk, qc = s.quad_keys, s.quad_counts
K = 20.0


def cond_counts(src, d, q):
    out = np.zeros(V)
    if d == 1:
        key = src * V + q
        lo, hi = np.searchsorted(tk, key * V), np.searchsorted(tk, key * V + V)
        out[tk[lo:hi] - key * V] = tc[lo:hi]
    elif d == 2:
        base = src * V ** 3
        lo, hi = np.searchsorted(qk, base), np.searchsorted(qk, base + V ** 3)
        ks, cs = np.asarray(qk[lo:hi]) - base, np.asarray(qc[lo:hi])
        m = (ks // V) % V == q
        np.add.at(out, ks[m] % V, cs[m])
    return out
log_uni = np.asarray(s.log_uni, np.float64)

pairs = []
usum, wsum = np.zeros(V), 0.0
for w in W:
    r = R.run(w, mc)
    t, a = r["t"], r["a3"]
    u = R.source_votes(r) / r["sdf"].mean()
    ea = np.tril(a, -1)
    usum += (ea.sum(0)[:, None] * u).sum(0)
    wsum += ea.sum()
    pw = np.exp(r["lp"])
    for i in range(1, T):
        for j in np.nonzero(a[i, :i] >= 0.3)[0]:
            pairs.append((u[j], pw[i], t[j], t[i], i - j, a[i, j]))
ubar = usum / wsum
print("pairs", len(pairs))

feat_names = ["mean", "pmi_match", "pmi_1", "pmi_2", "pmi_3", "pmi_4", "cond", "anti_q", "log_uni", "same_att",
              "same_tgt"]


def feats(src, q, d):
    f = {"mean": ubar, "pmi_match": pmi[min(d + 1, 8)][src], "anti_q": logf[q], "log_uni": log_uni,
         "same_att": (att == att[src]).astype(float), "same_tgt": (tgt == tgt[src]).astype(float)}
    for k in range(1, 5):
        f[f"pmi_{k}"] = pmi[k][src]
    if d <= 2:
        c = cond_counts(src, d, q)
        pc = (c + K * np.exp(logf[q])) / (c.sum() + K)
        f["cond"] = np.log(pc) - logf[q]
    else:
        f["cond"] = np.zeros(V)
    return f


X, Y, Wt = [], [], []
for u, pw, src, q, d, aij in pairs:
    f = feats(src, q, d)
    w = pw * aij
    x = np.stack([f[k] for k in feat_names], -1)
    x = x - (pw[:, None] * x).sum(0)
    y = u - (pw * u).sum()
    X.append(x)
    Y.append(y)
    Wt.append(w)
X, Y, Wt = np.stack(X), np.stack(Y), np.stack(Wt)


def r2(keys):
    idx = [feat_names.index(k) for k in keys]
    x = X[..., idx]
    xtx = np.einsum("nvk,nv,nvl->kl", x, Wt, x)
    xty = np.einsum("nvk,nv,nv->k", x, Wt, Y)
    b = np.linalg.solve(xtx + 1e-9 * np.eye(len(idx)), xty)
    sst = (Wt * Y * Y).sum()
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), [round(float(z), 2) for z in b]


res = {}
for keys in (["mean"], ["pmi_match"], ["cond"], ["anti_q"], ["log_uni"], ["same_att"], ["same_tgt"],
             ["pmi_1", "pmi_2", "pmi_3", "pmi_4"], ["pmi_match", "anti_q", "log_uni"],
             ["pmi_match", "cond", "anti_q", "log_uni"], ["pmi_match", "same_att", "same_tgt", "log_uni"],
             ["mean", "pmi_match"], ["mean", "pmi_match", "cond", "anti_q", "same_att", "same_tgt"],
             feat_names):
    res["+".join(keys)] = r2(keys)
    print("+".join(keys), res["+".join(keys)])
json.dump(res, open(f"{OUT}/diag2.json", "w"), indent=1)

# a few sources: top/bottom of the vote vs PMI_2
for name in [" Mom", " Lily", " run", " big", " what", " name", " lot", " fr", " happ", " From", " it", ".", " the"]:
    from program.vocab import ID
    sid = ID[name]
    o = (R.net.Ehat[sid] / 1.0) @ R.Wv.T
    o = o @ R.WO.T
    # use the vote of the first occurrence in the pairs if present, else skip
    us = [p[0] for p in pairs if p[2] == sid]
    if not us:
        continue
    uu = np.mean(us, 0) - ubar
    top, bot = np.argsort(-uu)[:10], np.argsort(uu)[:10]
    pt = np.argsort(-pmi[2][sid])[:10]
    print(repr(name), len(us), "| vote-mean top:", [S[v] for v in top], "| bottom:", [S[v] for v in bot],
          "| PMI2 top:", [S[v] for v in pt])
