"""Step 8: per-source vote u_j vs substitution and within-word tables (pairs a_ij >= 0.3, j < i), by query group.
Same pooled weighted R^2 as diag4."""
import json
import numpy as np
from h3lib import H3Run, T, V, OUT, class_ids
import h3rule
from analysis.fitlib import train_windows
from analysis.mech_eval import stats

s = stats()
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
tab = {k: np.load(f"{OUT}/{k}.npy").astype(np.float64) for k in ("sub_left", "sub_right", "word_pmi")}
starts_word = np.load(f"{OUT}/word_counts.npy").sum(1) > 0
G = h3rule.group_of()
R = H3Run()
W = train_windows(12, seed=11)[:, :T]
mc = R.mean_ctx(W)
X, Y, Wt, GG, names = [], [], [], [], None
for w in W:
    r = R.run(w, mc)
    t, a = r["t"], r["a3"]
    u = R.source_votes(r) / r["sdf"].mean()
    pw = np.exp(r["lp"])
    for i in range(1, T):
        for j in np.nonzero(a[i, :i] >= 0.3)[0]:
            d, src = i - j, t[j]
            f = {"pmi_match": pmi[min(d + 1, 8)][src], "pmi_1": pmi[1][src], "pmi_2": pmi[2][src],
                 "sub_left": tab["sub_left"][src], "sub_right": tab["sub_right"][src],
                 "word": tab["word_pmi"][src] * s.piece[t[i]]}
            names = list(f)
            x = np.stack([f[q] for q in names], -1)
            X.append(x - (pw[i][:, None] * x).sum(0))
            Y.append(u[j] - (pw[i] * u[j]).sum())
            Wt.append(pw[i] * a[i, j])
            GG.append(h3rule.GNAMES[G[t[i]]])
X, Y, Wt, GG = np.stack(X), np.stack(Y), np.stack(Wt), np.asarray(GG)


def r2(keys, m):
    idx = [names.index(q) for q in keys]
    x, y, w = X[m][..., idx], Y[m], Wt[m]
    xtx = np.einsum("nvk,nv,nvl->kl", x, w, x)
    xty = np.einsum("nvk,nv,nv->k", x, w, y)
    b = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    sst = (w * y * y).sum()
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), [round(float(z), 2) for z in b]


res = {}
for g in ["all"] + h3rule.GNAMES:
    m = np.ones(len(GG), bool) if g == "all" else GG == g
    if m.sum() < 10:
        continue
    row = {"n": int(m.sum())}
    for keys in (["pmi_match"], ["sub_left"], ["sub_right"], ["word"], ["sub_left", "sub_right"],
                 ["pmi_match", "sub_left", "sub_right"], ["pmi_match", "word"], ["pmi_1", "pmi_2", "sub_left", "sub_right", "word"],
                 ["pmi_match", "pmi_1", "pmi_2", "sub_left", "sub_right", "word"]):
        row["+".join(keys)] = r2(keys, m)
    res[g] = row
    print(g, row, flush=True)
json.dump(res, open(f"{OUT}/diag5.json", "w"), indent=1)
