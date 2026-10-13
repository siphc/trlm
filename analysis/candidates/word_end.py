"""Candidate WORD END: inside a split word, the word start votes for what follows finished words that begin with it.

At positions where the current token is a piece of a word that started L-1 >= 1 tokens back (word start t_j0):

    vote(v) = WEND-PMI(t_j0, v) = log((count(word starting with t_j0 is followed by v) + 44 P(v)) / (count + 44)) - log P(v)

counted over the TRAIN stream (analysis/investigations/word/wordpairs.py, table WEND; 44 = the Dirichlet strength of
corpus.py). Motivation (open-track insight): heads 3 and 0 read the word start from inside the word with weight ~0.25
each, and what they write when reading a word start correlates with WEND (+0.28, +0.30). One strength.
"""
import os

import numpy as np

NAME = "word_end"
HEADS = [0, 3]
_T = {}


def prepare(stats, corpus):
    from analysis.investigations.word import wordstats as WS

    C = np.load(os.path.join(WS.OUT, "WEND.npy")).astype(np.float64)
    p = np.asarray(stats.p_uni)
    _T["wend"] = (np.log((C + 44.0 * p[None, :]) / (C.sum(1, keepdims=True) + 44.0)) - np.log(p)[None, :]).astype(np.float32)
    _T["piece"] = WS.piece_flags()


def features(P, tokens):
    from analysis.investigations.word import wordstats as WS

    if not _T:
        prepare(P.STATS, None)
    t = np.asarray(tokens, dtype=np.int64)
    n = len(t)
    f = np.zeros((n, 2000))
    for i in range(n):
        L, _ = WS.context_key(t, i, _T["piece"])
        if L >= 2:
            f[i] = _T["wend"][t[i - L + 1]]
    return {"wend": f}
