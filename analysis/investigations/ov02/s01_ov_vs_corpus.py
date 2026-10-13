"""s01: effective OV maps of heads 0 and 2 vs corpus statistics (insight only).

Candidates compared with OV rows:
  PMI_k            corpus skip-gram PMI (program tables), k = 1, 2, 3
  RES_d            PMI_{d+1}(s, v) - E_{b ~ P_d(b|s)} PMI_1(b, v): what s says about the token d+1 later beyond what the
                   token d later (the current token) already says, if both were additive (closed form, no trigrams)
  CS1_K            conditional skip-gram at d = 1 from trigram counts:
                   sum_b P(b|s) [log P3(v|s,b) - log P1(v|b)] = sum_b P(b|s) log(1 + n(s,b,v) / (K P1(v|b))) + row const
  DP               direct path row of s (final LN at typical scale), for the anti-copy / complement check
"""
import json
import os
import time

import numpy as np

from common import OUT, V, global_r2, ov_cached, rowcorr

t0 = time.time()
from analysis.mech_eval import stats

S_ = stats()
OV = ov_cached().astype(np.float64)
uni = np.asarray(S_.unigram)
pu = np.asarray(S_.p_uni)
log_uni = np.log(pu)
pmi = {k: np.asarray(S_.pmi[k], np.float64) for k in range(1, 9)}
big = np.asarray(S_.bigram_counts)
T1 = big / np.maximum(big.sum(1, keepdims=True), 1)


def cs1(K):
    path = os.path.join(OUT, f"cs1_K{int(K)}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float64)
    keys = np.asarray(S_.tri_keys)
    cnt = np.asarray(S_.tri_counts).astype(np.float64)
    ab = keys // V
    a = ab // V
    b = ab % V
    c = keys % V
    n_ab = np.bincount(ab, weights=cnt, minlength=V * V)
    n_a = np.bincount(a, weights=cnt, minlength=V)
    p1 = np.exp(np.asarray(S_.log_follow, np.float64))
    w = n_ab[ab] / np.maximum(n_a[a], 1)
    val = w * np.log1p(cnt / (K * p1[b, c]))
    out = np.bincount(a * V + c, weights=val, minlength=V * V).reshape(V, V)
    out -= out.mean(1, keepdims=True)
    np.save(path, out.astype(np.float32))
    return out


def res(d):
    # P_d(b | s) from the smoothed skip-gram table (d = 1: raw bigram)
    Td = T1 if d == 1 else np.exp(pmi[d] + log_uni[None, :])
    Td = Td / Td.sum(1, keepdims=True)
    return pmi[d + 1] - Td @ pmi[1]


tabs = {"PMI_1": pmi[1], "PMI_2": pmi[2], "PMI_3": pmi[3], "RES_1": res(1), "RES_2": res(2),
        "CS1_K30": cs1(30.0), "CS1_K300": cs1(300.0), "CS1_K3": cs1(3.0)}
print("tables built %.0fs" % (time.time() - t0), flush=True)

freq = np.where(uni >= 20000)[0]
ws_f = uni[freq] / uni[freq].sum()
wv_u = np.full(V, 1.0 / V)
wv_p = pu / pu.sum()
res_out = {"n_rows": int(len(freq))}
for h in range(4):
    r = {}
    for name, X in tabs.items():
        r[name] = {"rowcorr_uniform_v": float(ws_f @ rowcorr(OV[h][freq], X[freq])),
                   "rowcorr_puni_v": float(ws_f @ rowcorr(OV[h][freq], X[freq], wv_p))}
    res_out[f"head{h}"] = r
    print(f"head {h}: " + "  ".join(f"{k} {v['rowcorr_uniform_v']:.3f}/{v['rowcorr_puni_v']:.3f}" for k, v in r.items()),
          flush=True)
# OV of different heads vs each other
cross = {}
for h in range(4):
    for g in range(h + 1, 4):
        cross[f"{h}-{g}"] = float(ws_f @ rowcorr(OV[h][freq], OV[g][freq]))
res_out["ov_cross_head_rowcorr"] = cross
print("cross-head OV row corr:", {k: round(v, 3) for k, v in cross.items()})
# global double-centred R^2 (rows weighted by frequency over all tokens, v uniform or by p_uni)
ws_all = uni / uni.sum()
for h in (0, 2):
    for wvname, wv in (("uniform_v", wv_u), ("puni_v", wv_p)):
        for combo in (["PMI_2"], ["PMI_1", "PMI_2", "PMI_3"], ["RES_1"], ["CS1_K30"], ["CS1_K300"], ["PMI_2", "CS1_K30"],
                      ["PMI_1", "PMI_2", "PMI_3", "CS1_K30", "RES_1"]):
            r2, c = global_r2(OV[h], [tabs[k] for k in combo], ws_all, wv)
            res_out.setdefault(f"head{h}_global_r2_{wvname}", {})["+".join(combo)] = [r2, [float(x) for x in c]]
            print(f"head {h} {wvname} R2 {'+'.join(combo)}: {r2:.3f} coef {np.round(c, 2)}", flush=True)
json.dump(res_out, open(os.path.join(OUT, "s01.json"), "w"), indent=1)
print("done %.0fs" % (time.time() - t0))
