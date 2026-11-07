"""Shared pieces of the ladder tools (logs/ladder/design.md): models, accounts, KL.

An account is a Python file exposing
    predict(X: np.ndarray) -> np.ndarray      X (n, T) int symbols; returns (n, T, V) log-probs, row t predicting the
                                              symbol after X[:, : t + 1]
and optionally setup(train_stream), called once with the train symbol stream (corpus statistics are free, as the
main track's decision 7). Its numbers must be written in its source; analysis/ladder/length.py checks the rules.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess

import numpy as np
import torch

from src.train import data_char
from src.train.ladder_runs import ROOT, RUNGS
from src.train.train import load_model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = range(5)


def model_dir(rung: str, arm: str, seed: int) -> str:
    return os.path.join(REPO, ROOT, f"{rung}_{arm}_s{seed}")


def load(rung: str, arm: str, seed: int):
    return load_model(os.path.join(model_dir(rung, arm, seed), "final.pt"))


def ctx(rung: str) -> int:
    return RUNGS[rung]["context_length"]


def windows(rung: str, split: str = "dev_full") -> np.ndarray:
    return data_char.load_eval(split, ctx(rung))


@torch.no_grad()
def model_logprobs(model, W: np.ndarray, bs: int = 200) -> np.ndarray:
    """fp32 log-probs (n, T, V) of the model on the inputs W[:, :-1]."""
    out = []
    for i in range(0, len(W), bs):
        x = torch.as_tensor(np.asarray(W[i:i + bs, :-1], np.int64))
        out.append(torch.log_softmax(model(x).float(), -1).numpy())
    return np.concatenate(out)


def load_account(path: str):
    spec = importlib.util.spec_from_file_location(f"account_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "setup"):
        mod.setup(data_char.load_stream("train"))
    return mod


def account_logprobs(mod, W: np.ndarray, bs: int = 200) -> np.ndarray:
    out = []
    for i in range(0, len(W), bs):
        x = np.asarray(W[i:i + bs, :-1], np.int64)
        y = np.asarray(mod.predict(x), np.float64)
        assert y.shape == x.shape + (data_char.vocab_size(),), y.shape
        assert np.allclose(np.logaddexp.reduce(y, -1), 0, atol=1e-6), "rows must be log-distributions"
        out.append(y)
    return np.concatenate(out)


def kl_pos(lp, lq) -> np.ndarray:
    """KL(p || q) per position, (n, T), in nats."""
    lp = np.asarray(lp, np.float64)
    lq = np.asarray(lq, np.float64)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def log_mean_exp(lps) -> np.ndarray:
    return np.log(np.mean([np.exp(np.asarray(lp, np.float64)) for lp in lps], 0))


def stat(x) -> dict:
    """Mean and standard error over windows of a per-position array (n, T) or per-window array (n,)."""
    x = np.asarray(x, np.float64)
    w = x.mean(-1) if x.ndim == 2 else x
    return {"mean": float(w.mean()), "se": float(w.std(ddof=1) / np.sqrt(len(w)))}


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip() or "none"
    except Exception:
        return "none"
