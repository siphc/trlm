"""ov02 candidate (compact form of ov02_earlier_votes, 4 strengths): what LOCAL-0 and LOCAL-2 write from EARLIER tokens
beyond the distance-matched skip-gram.

  LOCAL-0, from each earlier token s it reads (program attention, j < i):
    bigram  PMI_1(s, v): s's usual successors (head 0's map is shared with its current-token part, whose job is the bigram)
    alike   KPRED(s, v) for v != s: tokens that fill the same slot as s (sum_a sqrt(P(prev=a|s) P(prev=a|v))); expected
            negative: do not follow with a word like the one just read
  LOCAL-2, from each earlier token s it reads:
    copy    one-hot(s); expected negative: do not repeat s
    word3   WPMI_3(s, v): v is the first token of the 3rd unit after the unit of s (word-level skip-gram; pieces join
            the unit before them; no end-of-story token strictly between), PMI as in corpus.py
Tables: analysis/investigations/ov02/s03_word_tables.py (closed-form counts of the TRAIN stream).
"""
import os

import numpy as np

NAME = "ov02_earlier_votes4"
HEADS = [0, 2]
OUT = "/Users/svi/Desktop/trlm/analysis/out/candidates/ov02"
_T = {}


def features(P, tokens):
    from program import head0, head2
    if not _T:
        _T["bigram"] = np.asarray(P.STATS.pmi[1], np.float64)
        k = np.load(os.path.join(OUT, "kpred.npy")).astype(np.float64)
        np.fill_diagonal(k, 0.0)
        _T["alike"] = k
        _T["word3"] = np.load(os.path.join(OUT, "wpmi_3.npy")).astype(np.float64)
    n = len(tokens)
    a0 = head0.attention(tokens, P.STATS) * (1.0 - np.eye(n))
    a2 = head2.attention(tokens, P.STATS) * (1.0 - np.eye(n))
    c = np.zeros((n, 2000))
    np.add.at(c, (np.repeat(np.arange(n), n), np.tile(tokens, n)), a2.reshape(-1))
    return {"h0.bigram": a0 @ _T["bigram"][tokens], "h0.alike": a0 @ _T["alike"][tokens],
            "h2.copy": c, "h2.word3": a2 @ _T["word3"][tokens]}
