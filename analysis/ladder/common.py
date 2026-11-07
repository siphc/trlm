"""Shared pieces of the ladder tools: models, eval windows, KL, and the split of a one-block attention-only model
into its direct path x0 and its heads' outputs z (parts, finish, resampled).

The ladder's account tools (accounts, swap and replacement tests, behaviour tests, size checks) were removed with
decision 23; they are at git tag pre-proofs.
"""
from __future__ import annotations

import hashlib
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


@torch.no_grad()
def parts(model, X: np.ndarray):
    """x0 (n, T, d) and z (n, T, H, dh) as float64 arrays, for a one-block attention-only model (use it in float64)."""
    x = torch.as_tensor(X, dtype=torch.int64)
    n, T = x.shape
    blk = model.blocks[0]
    H, dh, d = model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(d, dim=-1)
    q, k, v = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k, v))
    z = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    return x0.double().numpy(), z.double().numpy()


@torch.no_grad()
def finish(model, x0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual sum x0 + output projection of z, final LayerNorm, unembedding: log-probs (n, T, V)."""
    n, T = x0.shape[:2]
    h = torch.as_tensor(x0) + model.blocks[0].attn.out(torch.as_tensor(z).reshape(n, T, -1))
    return torch.log_softmax(model.ln_f(h) @ model.wte.weight.T, -1).numpy()


def resampled(model, x0: np.ndarray, z: np.ndarray, site: str, rng: np.random.Generator) -> np.ndarray:
    """Log-probs with one site ("direct" or "head<h>") replaced by its value at a random other position."""
    n, T = x0.shape[:2]
    N = n * T
    d = rng.permutation(N)
    x0d, zd = x0.reshape(N, -1).copy(), z.reshape(N, z.shape[2], -1).copy()
    if site == "direct":
        x0d = x0d[d]
    else:
        h = int(site[4:])
        zd[:, h] = zd[d, h]
    return finish(model, x0d.reshape(x0.shape), zd.reshape(z.shape))


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
