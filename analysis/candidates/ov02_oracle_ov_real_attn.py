"""INSIGHT ONLY (uses model weights, never adoptable): the exact OV tables of heads 0 and 2 (position-averaged,
final LN at typical scale) read under the MODEL's real attention of heads 0 and 2, earlier tokens only.
Gives the ceiling for any vote statistic under the current attention rules."""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_oracle_ov_real_attn"
HEADS = [0, 2]
REPLACES = ["head0.near", "head0.far", "head2.near", "head2.far"]
_OV = None


def features(P, tokens):
    global _OV
    from common import model_attention, ov_cached
    if _OV is None:
        _OV = ov_cached().astype(np.float64)
    n = len(tokens)
    out = {}
    A = model_attention(tokens)
    for h in (0, 2):
        a = A[h] * (1.0 - np.eye(n))
        out[f"h{h}"] = a @ _OV[h][tokens]
    return out
