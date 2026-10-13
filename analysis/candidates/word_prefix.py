"""Candidate WORD (prefix): inside a split word, vote for the corpus continuations of the whole word so far.

Word so far at i: t_j0 .. t_i, t_j0 the latest token that is not a word piece (piece = text starts with a lowercase
letter); L = i - j0 + 1. At positions with L >= 2 (the current token is a piece of a word that started earlier):

    vote(v) = log Pw(v | word so far) - log P1(v | t_i),
    Pw(v) = (count(word so far, v) + K * P1(v | t_i)) / (count(word so far) + K),  K = 30

counted over the TRAIN stream (last min(L, 4) tokens of the word as the key; analysis/investigations/word/wordstats.py).
Zero elsewhere. Two strengths: L = 2 and L >= 3.
"""
import numpy as np

NAME = "word_prefix"
HEADS = [0, 2, 3]
K = 30.0
_T = {}


def prepare(stats, corpus):
    from analysis.investigations.word import wordstats as WS

    _T["piece"] = WS.piece_flags()
    _T["wk"], _T["wc"], _, _ = WS.load()


def features(P, tokens):
    from analysis.investigations.word import wordstats as WS
    from program.corpus import continuations

    if not _T:
        prepare(None, None)
    s = P.STATS
    n = len(tokens)
    f2, f3 = np.zeros((n, 2000)), np.zeros((n, 2000))
    for i in range(n):
        L, key = WS.context_key(tokens, i, _T["piece"])
        if L < 2:
            continue
        nxt, cnt = continuations(_T["wk"], _T["wc"], key)
        lp1 = s.log_follow[tokens[i]].astype(np.float64)
        p = K * np.exp(lp1) / (cnt.sum() + K)
        p[nxt] += cnt / (cnt.sum() + K)
        (f2 if L == 2 else f3)[i] = np.log(p) - lp1
    return {"L2": f2, "L3plus": f3}
