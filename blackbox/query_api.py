"""Black-box track: the only door to the model. Logits only.

    from blackbox.query_api import query_logits
    logits = query_logits([[0, 12, 345], [0, 7]])   # list of float32 arrays, shape (len_i, 2000)

The model id comes from TRLM_MODEL (default attn_s0). Sequences may be any length 1..256 and may be
synthetic. Row i of each result is the next-token logits after tokens[: i + 1].

Note: this module loads the checkpoint from models/ itself. In the black-box session the agent must
not read models/ directly; when the track is set up it should run this behind a separate process.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@lru_cache(maxsize=4)
def _model(model_id: str):
    sys.path.insert(0, _REPO)
    from harness.common import load_model

    return load_model(model_id)


def query_logits(batch_of_token_lists, model_id: str | None = None) -> list[np.ndarray]:
    import torch

    m = _model(model_id or os.environ.get("TRLM_MODEL", "attn_s0"))
    out = []
    with torch.no_grad():
        for toks in batch_of_token_lists:
            toks = [int(t) for t in toks]
            if not 1 <= len(toks) <= 256 or min(toks) < 0 or max(toks) >= 2000:
                raise ValueError("each sequence needs 1..256 tokens with ids in [0, 2000)")
            out.append(m(torch.tensor([toks]))[0].float().numpy())
    return out
