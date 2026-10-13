"""Fitting framework for mechanism programs ("explain, don't fit"; logs/decisions.md items 5-7).

A mechanism program scores next tokens as a weighted sum of mechanism votes,

    score_i[v] = sum_m strength_m * votes_m(tokens, i, v)        log q_i = log_softmax(score_i)

where each mechanism's votes come from corpus statistics (computed by program/corpus.py, exactly as the program
computes them in setup()) and a few named scalars. Only those scalars are fitted to the model (KL(model || program)
on TRAIN windows). The torch mechanisms here must mirror the NumPy program; check with `compare_with_program`.

  stats = corpus_stats(max_distance=8)       # cached in analysis/out/corpus_stats_d8.npz
  model = MechanismModel({"follow": Follow(stats), ...})
  fit_scalars(model, "attn_s0", steps=600)
  ledger(model, "attn_s0")                   # KL with every mechanism, and the rise when each one is switched off
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from analysis.fitlib import REPO, Target, train_windows

V, T = 2000, 256


def corpus_stats(max_distance: int = 8, with_repeats: bool = True) -> dict:
    """Unigram, distance-d pair counts (d = 1..max_distance) and per-story repeat counts of the TRAIN corpus,
    computed by program/corpus.py and cached as float32."""
    path = os.path.join(REPO, "analysis", "out", f"corpus_stats_d{max_distance}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from harness.common import load_corpus
    from program.corpus import Counts, story_repeats

    c = load_corpus()
    t0 = time.time()
    counts = Counts(c, max_distance=max_distance)
    out = {"unigram": counts.unigram.astype(np.float64), "n_tokens": np.array(counts.n_tokens)}
    for d in range(1, max_distance + 1):
        out[f"pairs{d}"] = counts.pair_counts(d).astype(np.float32)
    if with_repeats:
        once, twice = story_repeats(c)
        out["stories_with"], out["stories_with_twice"] = once, twice
    np.savez(path, **out)
    print(f"corpus statistics computed in {time.time() - t0:.0f}s -> {path}")
    return out


class MechanismModel(nn.Module):
    """log_softmax(sum_m strength_m * votes_m). Mechanisms are nn.Modules with forward(x) -> (B, T, V) votes."""

    def __init__(self, mechanisms: dict):
        super().__init__()
        self.mechanisms = nn.ModuleDict(mechanisms)
        self.strength = nn.ParameterDict({k: nn.Parameter(torch.tensor(1.0)) for k in mechanisms})
        self.off: set = set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = 0.0
        for name, mech in self.mechanisms.items():
            if name not in self.off:
                score = score + self.strength[name] * mech(x)
        return torch.log_softmax(score, -1)

    def scalars(self) -> dict:
        return {n: float(p.detach().cpu()) if p.numel() == 1 else p.detach().cpu().numpy().round(4).tolist()
                for n, p in self.named_parameters()}

    def n_scalars(self) -> int:
        return sum(p.numel() for p in self.parameters())


def kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    return (p_log.exp() * (p_log - q_log)).sum(-1).mean()


def fit_scalars(model: MechanismModel, model_id: str = "attn_s0", steps: int = 600, bs: int = 8, lr: float = 0.02,
                n_windows: int = 4000, device: str = "cpu", log_every: int = 100, seed: int = 0) -> list:
    tgt = Target(model_id, device)
    W = torch.as_tensor(train_windows(n_windows))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.time()
    for step in range(steps):
        x = W[torch.randint(0, len(W), (bs,), generator=g), :T].to(device)
        loss = kl(tgt.logprobs(x), model(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append((step, float(loss)))
            print(f"step {step} kl {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return hist


@torch.no_grad()
def holdout_kl(model: MechanismModel, model_id: str = "attn_s0", n: int = 64, device: str = "cpu", bs: int = 8) -> float:
    tgt = Target(model_id, device)
    W = torch.as_tensor(train_windows(200, seed=2000))[:n]
    tot = 0.0
    for i in range(0, len(W), bs):
        x = W[i : i + bs, :T].to(device)
        tot += float(kl(tgt.logprobs(x), model(x))) * len(x)
    return tot / len(W)


@torch.no_grad()
def ledger(model: MechanismModel, model_id: str = "attn_s0", n: int = 64, device: str = "cpu") -> dict:
    """KL of the full program and the KL rise when each mechanism alone is switched off."""
    full = holdout_kl(model, model_id, n, device)
    out = {"all": full}
    for name in model.mechanisms:
        model.off = {name}
        out[f"without {name}"] = holdout_kl(model, model_id, n, device) - full
    model.off = set()
    return out
