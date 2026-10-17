"""goal2 candidate: FOLLOW's and PHRASE's strengths depend on the current token's shape (word start, continuation
piece, other, end of story). REPLACES follow.bigram, phrase.trigram, phrase.fourgram with 4 shape-gated copies each.
"""
import numpy as np

NAME = "shape_gates"
HEADS = []
REPLACES = ["follow.bigram", "phrase.trigram", "phrase.fourgram"]
SHAPES = ["wordstart", "cont", "other", "eos"]


def features(P, tokens):
    from program import phrase
    s = P.STATS
    shape = s.shape[tokens]
    big = np.asarray(s.log_follow[tokens], np.float64)
    tri, quad = phrase.parts(tokens, s)
    out = {}
    for k, name in enumerate(SHAPES):
        g = (shape == k)[:, None]
        out[f"bigram.{name}.self"] = big * g
        out[f"trigram.{name}.self"] = tri * g
        out[f"fourgram.{name}.self"] = quad * g
    return out
