"""INSIGHT ONLY (uses model weights, never adoptable): the exact OV tables of heads 0 and 2 (position-averaged,
final LN at typical scale) read under the PROGRAM's attention rules for heads 0 and 2, earlier tokens only.
Gives the ceiling for any vote statistic under the current attention rules."""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_oracle_ov_prog_attn"
HEADS = [0, 2]
REPLACES = ["head0.near", "head0.far", "head2.near", "head2.far"]
_OV = None


def features(P, tokens):
    global _OV
    from common import ov_cached
    from program import head0, head2
    if _OV is None:
        _OV = ov_cached().astype(np.float64)
    n = len(tokens)
    out = {}
    for h, mod in ((0, head0), (2, head2)):
        a = mod.attention(tokens, P.STATS) * (1.0 - np.eye(n))
        out[f"h{h}"] = a @ _OV[h][tokens]
    return out
