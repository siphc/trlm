"""Fitting utilities for the open track.

Program tables are fitted by regression against the MODEL's log-probs (never the data), on windows
drawn from the TRAIN split, so dev stays a clean measurement of fidelity. Everything here is torch;
the fitted numbers are then written into program/ as named, lint-clean tables.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.common import load_model  # noqa: E402
from src.train.data import load_stream, n_windows, window  # noqa: E402

FIT_SEED = 1000  # train windows for fitting are drawn with this seed


def train_windows(n: int, seed: int = FIT_SEED) -> np.ndarray:
    s = load_stream("train")
    idx = np.random.default_rng(seed).choice(n_windows(len(s)), size=n, replace=False)
    return np.stack([np.asarray(window(s, int(i))) for i in np.sort(idx)]).astype(np.int64)


def ensure_disk(min_free_gb: float = 4.0, path: str = REPO) -> None:
    """Refuse to write large artifacts when the disk is nearly full (student checkpoints are 160 MB each)."""
    import shutil

    free = shutil.disk_usage(path).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f"only {free:.1f} GB free on disk; not saving (need {min_free_gb} GB)")


def model_name(model_id: str) -> str:
    """Short name for output folders: 'attn_s0' stays, '.../budget200M_x/final.pt' -> 'budget200M_x'."""
    base = os.path.basename(model_id.rstrip("/"))
    if base.endswith(".pt"):
        base = os.path.basename(os.path.dirname(os.path.abspath(model_id)))
    return base


class Target:
    """The model as a fitting target: log-probs and intermediate quantities on a device."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model = load_model(model_id).to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def logprobs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.model(x.to(self.device)).float(), -1)


def kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Mean per-token KL(p || q) over all leading dims."""
    return (p_log.exp() * (p_log - q_log)).sum(-1).mean()
