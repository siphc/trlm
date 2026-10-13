"""Step 4: is head 3's vote for source s a distance-MIXED skip-gram, mixed by how far away s usually is when head 3
reads it (OV is query-independent, so it cannot use the actual distance)?

w_s(d): share of head 3's attention onto token s at distance d (d = 0..7, d >= 7 pooled), measured on 100 TRAIN
windows (seed 13, model attention; diagnostic only), backed off to the ATTENTION class for rare tokens.
  mix_log   sum_d w_s(d) PMI_{d+1}(s, v)
  mix_prob  log sum_d w_s(d) exp PMI_{d+1}(s, v)
  mix_noself  same as mix_log without d = 0 (earlier tokens only)
Pooled weighted R^2 of the per-source votes u_j (pairs a_ij >= 0.3, j < i, 12 windows seed 11), as in diag2.
"""
import json
import numpy as np
from h3lib import H3Run, T, V, OUT, class_ids, tok_strs
from analysis.fitlib import train_windows
from analysis.mech_eval import stats

s = stats()
att, _ = class_ids("ATTENTION")
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
R = H3Run()
Wd = train_windows(100, seed=13)[:, :T]
mc = R.mean_ctx(train_windows(12, seed=11)[:, :T])
D = 8
mass = np.zeros((V, D))
dist = np.arange(T)[:, None] - np.arange(T)[None, :]
db = np.minimum(dist, D - 1)
for w in Wd:
    r = R.net.run(w)
    a = r["pats"][3]
    t = r["t"]
    for d in range(D):
        m = (db == d) & (dist >= 0)
        np.add.at(mass[:, d], t, (a * m).sum(0))
cm = np.zeros((att.max() + 1, D))
np.add.at(cm, att, mass)
prior = 5.0
wts = (mass + prior * cm[att] / cm[att].sum(1, keepdims=True)) / (mass.sum(1, keepdims=True) + prior)
np.save(f"{OUT}/w_sd_model.npy", wts)
P = np.stack([pmi[min(d + 1, 8)] for d in range(D)], 0)   # (D, V, V)


def mixes(src):
    w = wts[src]
    out = {"mix_log": np.tensordot(w, P[:, src], 1),
           "mix_prob": np.log(np.tensordot(w, np.exp(P[:, src]), 1)),
           "mix_noself": np.tensordot(w[1:] / w[1:].sum(), P[1:, src], 1)}
    return out


W = train_windows(12, seed=11)[:, :T]
X, Y, Wt, names = [], [], [], None
for w in W:
    r = R.run(w, mc)
    t, a = r["t"], r["a3"]
    u = R.source_votes(r) / r["sdf"].mean()
    pw = np.exp(r["lp"])
    for i in range(1, T):
        for j in np.nonzero(a[i, :i] >= 0.3)[0]:
            d = i - j
            f = {"pmi_match": pmi[min(d + 1, 8)][t[j]], **mixes(t[j])}
            names = list(f)
            x = np.stack([f[k] for k in names], -1)
            x = x - (pw[i][:, None] * x).sum(0)
            X.append(x)
            Y.append(u[j] - (pw[i] * u[j]).sum())
            Wt.append(pw[i] * a[i, j])
X, Y, Wt = np.stack(X), np.stack(Y), np.stack(Wt)
res = {}
for keys in (["pmi_match"], ["mix_log"], ["mix_prob"], ["mix_noself"], ["pmi_match", "mix_log"]):
    idx = [names.index(k) for k in keys]
    x = X[..., idx]
    xtx = np.einsum("nvk,nv,nvl->kl", x, Wt, x)
    xty = np.einsum("nvk,nv,nv->k", x, Wt, Y)
    b = np.linalg.solve(xtx, xty)
    sst = (Wt * Y * Y).sum()
    res["+".join(keys)] = (round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), [round(float(z), 2) for z in b])
    print(keys, res["+".join(keys)])
json.dump(res, open(f"{OUT}/diag3.json", "w"), indent=1)
