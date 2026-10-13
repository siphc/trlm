"""INSIGHT ONLY (model weights): exact OV of heads 0 and 2 truncated to its top OV02_RANK singular components
(rows weighted by sqrt token frequency, columns by sqrt unigram probability), read under the model's real attention
(OV02_ATTN=real) or the program's (prog), earlier tokens only. How many directions do the heads really write?"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_oracle_lowrank"
HEADS = [0, 2]
REPLACES = ["head0.near", "head0.far", "head2.near", "head2.far"]
RANK = int(os.environ.get("OV02_RANK", "4"))
ATTN = os.environ.get("OV02_ATTN", "real")
_OV = None


def lowrank(P):
    from common import ov_cached
    OV = ov_cached().astype(np.float64)
    uni = np.asarray(P.STATS.unigram)
    wr = np.sqrt(uni / uni.sum())[:, None] + 1e-6
    wc = np.sqrt(np.asarray(P.STATS.p_uni))[None, :]
    out = {}
    for h in (0, 2):
        U, s, Vt = np.linalg.svd(OV[h] * wr * wc, full_matrices=False)
        out[h] = ((U[:, :RANK] * s[:RANK]) @ Vt[:RANK]) / wr / wc
    return out


def features(P, tokens):
    global _OV
    from program import head0, head2
    if _OV is None:
        _OV = lowrank(P)
    n = len(tokens)
    if ATTN == "real":
        from common import model_attention
        A = model_attention(tokens)
        att = {0: A[0], 2: A[2]}
    else:
        att = {0: head0.attention(tokens, P.STATS), 2: head2.attention(tokens, P.STATS)}
    return {f"h{h}": (att[h] * (1.0 - np.eye(n))) @ _OV[h][tokens] for h in (0, 2)}
