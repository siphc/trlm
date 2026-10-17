"""goal2 candidate: FOLLOW responds to the corpus bigram piecewise-linearly instead of linearly.

    vote(v) = B0 * x + sum_k B_k * max(x - knot_k, 0),   x = log P1(v | current token),  knots -8, -6, -4, -2

Readable form: "the model trusts frequent continuations more (or less) than rare ones". REPLACES follow.bigram.
"""
import numpy as np

NAME = "follow_pwl"
HEADS = []
REPLACES = ["follow.bigram"]
KNOTS = [-8.0, -6.0, -4.0, -2.0]


def features(P, tokens):
    x = np.asarray(P.STATS.log_follow[tokens], np.float64)
    out = {"x.self": x}
    for k in KNOTS:
        out[f"hinge{int(k)}.self"] = np.maximum(x - k, 0.0)
    return out
