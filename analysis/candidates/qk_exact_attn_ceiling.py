"""DIAGNOSTIC ONLY (uses the model's weights; never adoptable): LOCAL-0 and LOCAL-2 read with the model's exact
head 0 / head 2 attention patterns, with the program's skip-gram votes. The ceiling of any attention-rule fix,
pair-specific or not, under the current vote rule."""
import os

import numpy as np

NAME = "exact_attn_ceiling"
HEADS = [int(h) for h in os.environ.get("QK_CEIL_HEADS", "0,2").split(",")]
REPLACES = ["head0.self", "head0.near", "head0.far", "head2.self", "head2.near", "head2.far"]
_M = None
_HEADS = [int(h) for h in os.environ.get("QK_CEIL_HEADS", "0,2").split(",")]


def features(P, tokens):
    global _M
    from analysis.tools import Model
    from program import head0, head2
    from program.readers import skipgram_votes

    if _M is None:
        _M = Model("attn_s0")
    t = np.asarray(tokens, dtype=np.int64)
    pats = _M.run(t)["patterns"]
    out = {}
    for h, mod in ((0, head0), (2, head2)):
        a = pats[h] if h in _HEADS else mod.attention(t, P.STATS)
        for part, unit in (("self", (1, 0, 0)), ("near", (0, 1, 0)), ("far", (0, 0, 1))):
            out[f"h{h}.{part}"] = skipgram_votes(a, t, P.STATS, *unit)
    return out
