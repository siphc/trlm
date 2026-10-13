"""Helpers for claim tests in program/tests/. READ-ONLY for replacement agents.

The harness runs `pytest program/tests` with TRLM_MODEL and TRLM_PROGRAM set. Claim tests use:

    from harness import claims
    P = claims.program()                 # the replacement program module
    W = claims.claim_windows()           # (200, 257) held-out dev windows (not in dev_fast/dev_full)
    lp = claims.model_logprobs(W[:20])   # (20, 256, 2000) fp32 model log-probs (the target)
    lp_abl = claims.model_logprobs(W[:20], ablate_heads=[2])   # head 2 output zeroed (open track)
    lp_mean = claims.model_logprobs(W[:20], ablate_heads=[2], ablation="mean")      # head 2 output -> its mean
    lp_ctx = claims.model_logprobs(W[:20], ablate_heads=[2], ablation="earlier")    # only its earlier-token part
    claims.kl(lp, q)                     # per-token KL(model || program), shape (n, 256)
"""
from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache

import numpy as np

from harness.common import eval_windows, head_means, load_model
from harness.common import model_logprobs as _model_logprobs


def model_id() -> str:
    return os.environ["TRLM_MODEL"]


@lru_cache(maxsize=1)
def program():
    path = os.environ.get("TRLM_PROGRAM", os.path.join(os.getcwd(), "program"))
    parent, name = os.path.dirname(path), os.path.basename(path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(name)


def claim_windows() -> np.ndarray:
    return eval_windows("claims")


@lru_cache(maxsize=1)
def _model():
    return load_model(model_id())


@lru_cache(maxsize=1)
def _means():
    """Head means over all 200 claim windows (mean and earlier-token ablation)."""
    return head_means(_model(), claim_windows())


def model_logprobs(windows: np.ndarray, ablate_heads=(), ablation: str = "zero") -> np.ndarray:
    """Model log-probs; ablation "zero" | "mean" | "earlier" for the listed heads (harness.common)."""
    means = _means() if ablate_heads and ablation in ("mean", "earlier") else None
    return _model_logprobs(_model(), windows, ablate_heads=tuple(ablate_heads), ablation=ablation, means=means)


def kl(p_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    p = np.exp(p_log)
    with np.errstate(invalid="ignore"):
        return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)
