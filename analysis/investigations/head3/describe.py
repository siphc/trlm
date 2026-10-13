"""Step 1: where head 3's earlier-token effect sits, and examples.

TRAIN windows seed 11 (exploration). Effect y = lp - lpa (head 3 earlier part mean-ablated).
Reports by query ATTENTION class: positions, mean shift KL(model || ablated), share of total shift, attention mass
on self / d=1 / d=2-4 / d>=5, top attended key classes (earlier tokens). Prints the biggest-shift examples.
"""
import json
import sys

import numpy as np

from h3lib import H3Run, T, V, OUT, class_ids, kl_rows, tok_strs, wcorr

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 24
from analysis.fitlib import train_windows

W = train_windows(NW, seed=11)[:, :T]
R = H3Run()
mc = R.mean_ctx(W)
cls, names = class_ids("ATTENTION")
S = tok_strs()
C = len(names)
stat = {k: np.zeros(C) for k in ["n", "kl", "self", "d1", "d2_4", "d5p", "dla_r2_num", "dla_r2_den"]}
keymass = np.zeros((C, C))
examples = []
dist = np.arange(T)[:, None] - np.arange(T)[None, :]
tot_r2 = [0.0, 0.0]
for wi, w in enumerate(W):
    r = R.run(w, mc)
    t = r["t"]
    lp, lpa, a = r["lp"], r["lpa"], r["a3"]
    k = kl_rows(lp, lpa)
    q = cls[t]
    np.add.at(stat["n"], q, 1)
    np.add.at(stat["kl"], q, k)
    np.add.at(stat["self"], q, np.diag(a))
    for key, lo, hi in (("d1", 1, 1), ("d2_4", 2, 4), ("d5p", 5, 999)):
        np.add.at(stat[key], q, (a * ((dist >= lo) & (dist <= hi))).sum(1))
    ea = a * (dist >= 1)
    for c in range(C):
        rows = q == c
        if rows.any():
            keymass[c] += np.bincount(cls[t], weights=ea[rows].sum(0), minlength=C)
    # direct-logit approximation of the earlier effect: sum_{j<i} a_ij u_j / sdf_i - mean
    u = R.source_votes(r)
    mcu = ((mc @ R.WO.T) - (mc @ R.WO.T).mean()) * R.net.gf @ R.net.WE.T
    dla = (ea @ u - mcu[None, :]) / r["sdf"][:, None]
    y = lp - lpa
    pw = np.exp(lp)
    yc = y - (pw * y).sum(1, keepdims=True)
    dc = dla - (pw * dla).sum(1, keepdims=True)
    b = (pw * yc * dc).sum() / (pw * dc * dc).sum()
    tot_r2[0] += (pw * (yc - dc) ** 2).sum()
    tot_r2[1] += (pw * yc * yc).sum()
    for i in np.argsort(-k)[:3]:
        top_src = np.argsort(-ea[i])[:4]
        gain = np.where((np.exp(lp[i]) > 0.01) | (np.exp(lpa[i]) > 0.01), y[i], np.nan)
        up = [j for j in np.argsort(-np.nan_to_num(gain, nan=-99))[:6]]
        dn = [j for j in np.argsort(np.nan_to_num(gain, nan=99))[:6]]
        examples.append({
            "kl": round(float(k[i]), 3),
            "ctx": "".join(S[x] for x in t[max(0, i - 14): i + 1]),
            "query": S[t[i]], "qclass": names[q[i]], "self": round(float(a[i, i]), 2),
            "attends": [(S[t[j]], int(i - j), round(float(ea[i, j]), 2)) for j in top_src],
            "up": [(S[v], round(float(y[i, v]), 2), round(float(np.exp(lp[i, v])), 3)) for v in up],
            "down": [(S[v], round(float(y[i, v]), 2), round(float(np.exp(lpa[i, v])), 3)) for v in dn],
        })
    print(wi, flush=True)

tot = stat["kl"].sum()
order = np.argsort(-stat["kl"])
rows = []
for c in order[:30]:
    n = stat["n"][c]
    km = keymass[c] / max(keymass[c].sum(), 1e-9)
    top = np.argsort(-km)[:4]
    rows.append({"class": names[c], "n": int(n), "mean_kl": round(stat["kl"][c] / n, 3),
                 "share": round(stat["kl"][c] / tot, 3),
                 "self": round(stat["self"][c] / n, 2), "d1": round(stat["d1"][c] / n, 2),
                 "d2_4": round(stat["d2_4"][c] / n, 2), "d5p": round(stat["d5p"][c] / n, 2),
                 "keys": [(names[x], round(float(km[x]), 2)) for x in top]})
for row in rows:
    print(row)
print("mean shift", tot / stat["n"].sum(), "DLA unscaled residual frac", tot_r2[0] / tot_r2[1])
examples.sort(key=lambda e: -e["kl"])
for e in examples[:40]:
    print(json.dumps(e, ensure_ascii=False, default=float))
json.dump({"by_class": rows, "examples": examples}, open(f"{OUT}/describe.json", "w"), indent=1, default=float)
