"""ov02 candidate: what LOCAL-0 and LOCAL-2 write from EARLIER tokens, beyond the distance-matched skip-gram.

For each head h in (0, 2) and each earlier token s = t_j it reads (program attention a_h(i, j), j < i), five votes:
  bigram  PMI_1(s, v): s's usual successors (the head's map is shared with the current-token part)
  slot    KPRED(s, v) = sum_a sqrt(P(prev = a | s) P(prev = a | v)): tokens that fill the same slot as s
          (the heads' maps are most negative on s itself and on words like it)
  copy    one-hot(s)
  word2   WPMI_2(s, v), word3 WPMI_3(s, v): word-level skip-grams, v is the first token of the 2nd / 3rd unit after the
          unit of s (units: a token that is not a word piece starts a new unit; pieces join the unit before them;
          no end-of-story token strictly between); smoothed and turned into PMI exactly like corpus.py's skip-grams
10 strengths. Counting rules: closed form from the TRAIN stream (analysis/investigations/ov02/s03_word_tables.py),
cached under analysis/out/candidates/ov02/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_earlier_votes"
HEADS = [0, 2]
OUT = "/Users/svi/Desktop/trlm/analysis/out/candidates/ov02"
_T = {}


def prepare(stats, corpus):
    if not all(os.path.exists(os.path.join(OUT, f)) for f in ("wpmi_2.npy", "wpmi_3.npy", "kpred.npy")):
        raise RuntimeError("run analysis/investigations/ov02/s03_word_tables.py first")


def features(P, tokens):
    from program import head0, head2
    if not _T:
        _T["bigram"] = np.asarray(P.STATS.pmi[1], np.float64)
        _T["slot"] = np.load(os.path.join(OUT, "kpred.npy")).astype(np.float64)
        _T["word2"] = np.load(os.path.join(OUT, "wpmi_2.npy")).astype(np.float64)
        _T["word3"] = np.load(os.path.join(OUT, "wpmi_3.npy")).astype(np.float64)
    n = len(tokens)
    out = {}
    for h, mod in ((0, head0), (2, head2)):
        a = mod.attention(tokens, P.STATS) * (1.0 - np.eye(n))
        for k, X in _T.items():
            out[f"h{h}.{k}"] = a @ X[tokens]
        c = np.zeros((n, 2000))
        np.add.at(c, (np.repeat(np.arange(n), n), np.tile(tokens, n)), a.reshape(-1))
        out[f"h{h}.copy"] = c
    return out
