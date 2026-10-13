"""Step 5: are head 3's votes class-level (the next token's TARGET / ATTENTION class) rather than token-level?

cls_pmi_k^C(s, v) = log P_k(C(v) | s) - log P(C(v)), with P_k(c | s) = sum_{v in c} P_k(v | s) from the program's
smoothed skip-gram tables (C = TARGET or ATTENTION classes); within = PMI_k(s, v) - cls_pmi_k(s, v) (the token-level
remainder inside the class). Pooled weighted R^2 of per-source votes u_j (pairs a_ij >= 0.3, j < i) overall and by
query group.
"""
import json
import numpy as np
from h3lib import H3Run, T, V, OUT, class_ids, tok_strs
from analysis.fitlib import train_windows
from analysis.mech_eval import stats

s = stats()
att, att_names = class_ids("ATTENTION")
tgt, _ = class_ids("TARGET")
pu = np.asarray(s.p_uni, np.float64)
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}


def class_pmi(k, C):
    nC = C.max() + 1
    M = np.zeros((V, nC)); M[np.arange(V), C] = 1
    pk = np.exp(pmi[k]) * pu[None, :]
    pk /= pk.sum(1, keepdims=True)
    pc = pu @ M
    return np.log(pk @ M) - np.log(pc)[None, :]


cp = {(k, nm): class_pmi(k, C) for k in (1, 2, 3) for nm, C in (("tgt", tgt), ("att", att))}
R = H3Run()
W = train_windows(12, seed=11)[:, :T]
mc = R.mean_ctx(W)
groups = {"LINKER": ["LINKERS"], "LINKVERB": ["LINKING_VERBS"], "BOUNDARY": ["SENTENCE_ENDS", "SENTENCE_STARTS"],
          "PIECE": [n for n in att_names if n.startswith(("FRAGMENTS", "SUFFIXES", "LETTERS", "ONSETS", "NAME_END"))]}
X, Y, Wt, G, names = [], [], [], [], None
for w in W:
    r = R.run(w, mc)
    t, a = r["t"], r["a3"]
    u = R.source_votes(r) / r["sdf"].mean()
    pw = np.exp(r["lp"])
    for i in range(1, T):
        for j in np.nonzero(a[i, :i] >= 0.3)[0]:
            d, src = i - j, t[j]
            k = min(d + 1, 3)
            f = {"pmi_match": pmi[min(d + 1, 8)][src], "pmi_2": pmi[2][src],
                 "tgt_match": cp[(k, "tgt")][src][tgt], "att_match": cp[(k, "att")][src][att],
                 "tgt_2": cp[(2, "tgt")][src][tgt], "att_2": cp[(2, "att")][src][att],
                 "same_att": (att == att[src]).astype(float)}
            f["within_tgt"] = f["pmi_match"] - f["tgt_match"]
            names = list(f)
            x = np.stack([f[q] for q in names], -1)
            X.append(x - (pw[i][:, None] * x).sum(0))
            Y.append(u[j] - (pw[i] * u[j]).sum())
            Wt.append(pw[i] * a[i, j])
            qn = att_names[att[t[i]]]
            G.append(next((g for g, l in groups.items() if qn in l), "OTHER"))
X, Y, Wt, G = np.stack(X), np.stack(Y), np.stack(Wt), np.asarray(G)


def r2(keys, m):
    idx = [names.index(q) for q in keys]
    x, y, w = X[m][..., idx], Y[m], Wt[m]
    xtx = np.einsum("nvk,nv,nvl->kl", x, w, x)
    xty = np.einsum("nvk,nv,nv->k", x, w, y)
    b = np.linalg.solve(xtx + 1e-9 * np.eye(len(idx)), xty)
    sst = (w * y * y).sum()
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3)


res = {}
for g in ["all", "LINKER", "LINKVERB", "BOUNDARY", "PIECE", "OTHER"]:
    m = np.ones(len(G), bool) if g == "all" else G == g
    row = {"n": int(m.sum())}
    for keys in (["pmi_match"], ["pmi_2"], ["tgt_match"], ["att_match"], ["tgt_2"], ["att_2"], ["same_att"],
                 ["tgt_match", "within_tgt"], ["pmi_match", "tgt_match", "att_match", "same_att"]):
        row["+".join(keys)] = r2(keys, m)
    res[g] = row
    print(g, row)
json.dump(res, open(f"{OUT}/diag4.json", "w"), indent=1)
