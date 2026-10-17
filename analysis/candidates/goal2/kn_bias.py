"""goal2 candidate: a background preference for tokens that follow many different tokens (Kneser-Ney continuation
probability), added to FOLLOW's unigram term.

    P_cont(v) = N1+(. v) / sum_u N1+(. u),   N1+(. v) = number of distinct tokens a with count(a, v) > 0 (TRAIN stream)
"""
import numpy as np

NAME = "kn_bias"
HEADS = []
REPLACES = []


def p_cont(s):
    n = (np.asarray(s.bigram_counts) > 0).sum(0).astype(np.float64) + 1.0
    return n / n.sum()


def features(P, tokens):
    lc = np.log(p_cont(P.STATS))
    return {"cont.self": np.broadcast_to(lc, (len(tokens), 2000)).copy()}
