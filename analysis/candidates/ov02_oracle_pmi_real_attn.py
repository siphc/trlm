"""INSIGHT ONLY (uses the model's attention, never adoptable): the program's skip-gram PMI votes (readers.py rule,
distance-matched) for heads 0 and 2, read under the MODEL's real attention of heads 0 and 2, earlier tokens only.
Separates the attention gap from the vote gap."""
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_oracle_pmi_real_attn"
HEADS = [0, 2]
REPLACES = ["head0.near", "head0.far", "head2.near", "head2.far"]


def features(P, tokens):
    from common import model_attention
    from program.readers import skipgram_votes
    n = len(tokens)
    A = model_attention(tokens)
    out = {}
    for h in (0, 2):
        a = A[h] * (1.0 - np.eye(n))
        out[f"h{h}.near"] = skipgram_votes(a, tokens, P.STATS, 0.0, 1.0, 0.0)
        out[f"h{h}.far"] = skipgram_votes(a, tokens, P.STATS, 0.0, 0.0, 1.0)
    return out
