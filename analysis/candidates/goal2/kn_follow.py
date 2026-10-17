"""goal2 candidate: FOLLOW as an interpolated Kneser-Ney bigram (Kneser & Ney 1995; Chen & Goodman 1998), with the
continuation probability as background, instead of FOLLOW's Dirichlet/class-blurred bigram and the unigram.

    P_KN(v | a) = max(c(a, v) - D, 0) / c(a) + D * N1+(a .) / c(a) * P_cont(v),   D = 0.75
    P_cont(v)   = N1+(. v) / sum_u N1+(. u)
REPLACES follow.bigram and follow.unigram with log P_KN(v | t_i) and log P_cont(v).
"""
import numpy as np

from kn_bias import p_cont

NAME = "kn_follow"
HEADS = []
REPLACES = ["follow.bigram", "follow.unigram"]
D = 0.75
_T = {}


def table(s):
    if "kn" not in _T:
        c = np.asarray(s.bigram_counts, np.float64)
        ca = np.maximum(c.sum(1, keepdims=True), 1.0)
        n1 = (c > 0).sum(1, keepdims=True)
        pc = p_cont(s)
        p = np.maximum(c - D, 0.0) / ca + (D * n1 / ca) * pc[None, :]
        p[c.sum(1) == 0] = pc
        _T["kn"] = np.log(p).astype(np.float32)
        _T["cont"] = np.log(pc)
    return _T["kn"], _T["cont"]


def features(P, tokens):
    kn, cont = table(P.STATS)
    return {"bigram.self": np.asarray(kn[tokens], np.float64),
            "cont.self": np.broadcast_to(cont, (len(tokens), 2000)).copy()}
