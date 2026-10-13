"""Step 2: which corpus statistic matches what head 3 writes from the token it reads?

For (query i, source j<i) pairs where head 3 puts attention >= 0.3 on an earlier token, compare the model's
per-source logit vote u_j (head 3's OV output for that source, through the final LN) with candidate corpus
statistics f(s=t_j, v, d=i-j, q=t_i), by the correlation over the vocabulary weighted by the model's next-token
probabilities at i (the tokens that matter there). Averages are weighted by a_ij.

Candidates
  pmi_match   PMI_{d+1}(s, v)  (v sits d+1 tokens after s: distance-matched skip-gram, the card's rule)
  pmi_2       PMI_2(s, v)
  bag         log mean_{k=1..8} exp PMI_k(s, v)   (v follows s somewhere in the next 8 tokens)
  topic       topic(s, v)
  copy        [v == s]
  cond        log P(v | s at -d, q) - log P1(v | q)   from trigram (d=1) / 4-gram (d=2) counts, 20 pseudo-counts
              (what s predicts beyond what the current token predicts; query-specific)
  cond_bigr   log P(v | s at -d, q)  (no baseline)
"""
import sys

import numpy as np

from h3lib import H3Run, T, V, OUT, class_ids, tok_strs, wcorr
from analysis.fitlib import train_windows
from analysis.mech_eval import stats

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 16
W = train_windows(NW, seed=11)[:, :T]
R = H3Run()
mc = R.mean_ctx(W)
s = stats()
cls, names = class_ids("ATTENTION")
S = tok_strs()
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
logf = np.asarray(s.log_follow, np.float64)
tk, tc = np.asarray(s.tri_keys), np.asarray(s.tri_counts)
qk, qc = s.quad_keys, s.quad_counts
K = 20.0


def cond_counts(src, d, q):
    """counts of v after (src at -d, q at 0) for d = 1, 2."""
    out = np.zeros(V)
    if d == 1:
        key = src * V + q
        lo, hi = np.searchsorted(tk, key * V), np.searchsorted(tk, key * V + V)
        out[tk[lo:hi] - key * V] = tc[lo:hi]
    elif d == 2:
        base = src * V ** 3
        lo, hi = np.searchsorted(qk, base), np.searchsorted(qk, base + V ** 3)
        ks, cs = np.asarray(qk[lo:hi]) - base, np.asarray(qc[lo:hi])
        c = (ks // V) % V
        m = c == q
        np.add.at(out, ks[m] % V, cs[m])
    return out


names_c = ["pmi_match", "pmi_2", "bag", "topic", "copy", "cond", "cond_bigr"]
acc = {k: [] for k in names_c}
wts, dd, qcl = [], [], []
for w in W:
    r = R.run(w, mc)
    t = r["t"]
    a = r["a3"]
    u = R.source_votes(r) / r["sdf"].mean()
    lp = r["lp"]
    for i in range(1, T):
        for j in np.nonzero(a[i, :i] >= 0.3)[0]:
            d = i - j
            src = t[j]
            pw = np.exp(lp[i])[None, :]
            feats = {
                "pmi_match": pmi[min(d + 1, 8)][src],
                "pmi_2": pmi[2][src],
                "bag": np.log(np.mean([np.exp(pmi[k][src]) for k in range(1, 9)], 0)),
                "topic": np.asarray(s.topic[src], np.float64),
                "copy": (np.arange(V) == src).astype(float),
            }
            if d <= 2:
                c = cond_counts(src, d, t[i])
                pc = (c + K * np.exp(logf[t[i]])) / (c.sum() + K)
                feats["cond"] = np.log(pc) - logf[t[i]]
                feats["cond_bigr"] = np.log(pc)
            for k in names_c:
                acc[k].append(float(wcorr(u[j][None], feats[k][None], pw)[0]) if k in feats else np.nan)
            wts.append(a[i, j])
            dd.append(d)
            qcl.append(cls[t[i]])
wts, dd, qcl = map(np.asarray, (wts, dd, qcl))
print("pairs", len(wts))
groups = {"all": np.ones(len(wts), bool), "d=1": dd == 1, "d=2": dd == 2, "d>=3": dd >= 3}
for cname in ["LINKERS", "LINKING_VERBS", "SENTENCE_ENDS", "FRAGMENTS_SHORT", "FRAGMENTS_MIXED", "FRAGMENTS_VC",
              "SENTENCE_STARTS", "DETERMINERS", "LETTERS_LOWER", "PREPOSITIONS"]:
    groups["q=" + cname] = qcl == names.index(cname)
res = {}
for g, m in groups.items():
    row = {"n": int(m.sum())}
    for k in names_c:
        x = np.asarray(acc[k])[m]
        ok = ~np.isnan(x)
        row[k] = round(float(np.average(x[ok], weights=wts[m][ok])), 3) if ok.any() else None
    res[g] = row
    print(g, row)
import json
json.dump(res, open(f"{OUT}/diag.json", "w"), indent=1)
