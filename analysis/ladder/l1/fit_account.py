"""Fit the L1 v2 account's constants to the model on a coarse grid (for P2).

The account's forward pass (analysis/ladder/l1/build_account.py: no position, one head) is rebuilt in torch. It is
initialised from the model's weights and fitted by KL(model || account) on train windows (sampler seed 2000 + seed;
dev is not used for fitting). EMBED is held on a grid of step `step` (straight-through rounding), and the other
constants are rounded to 2 decimals at the end. Returns the fitted constants in build_account's format.

    .venv/bin/python -m analysis.ladder.l1.fit_account --seed 0 --step 0.5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from analysis.ladder.common import kl_pos, load, model_logprobs, windows
from analysis.ladder.l1.build_account import constants
from src.train import data_char
from src.train.data import BatchSampler


class Account(torch.nn.Module):
    def __init__(self, c: dict, step: float):
        super().__init__()
        self.step = step
        self.ln = c["LAYERNORM"]
        for k in ("EMBED", "POS", "WQ", "WK", "WV", "WO", "OUT_GAIN"):
            setattr(self, k, torch.nn.Parameter(torch.as_tensor(np.array(c[k]), dtype=torch.float64)))

    def embed(self):
        E = self.EMBED
        if not self.step:
            return E
        return E + (torch.round(E / self.step) * self.step - E).detach()

    def norm(self, x):
        return F.layer_norm(x, (x.shape[-1],)) if self.ln else x

    def head(self, x):
        u = self.norm(self.embed()[x] + self.POS)
        T = x.shape[1]
        s = (u @ self.WQ.T) @ (u @ self.WK.T).transpose(1, 2)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        return s.softmax(-1) @ (u @ self.WV.T)

    def forward(self, x):
        E = self.embed()
        h = E[x] + self.POS
        return self.norm(h + self.head(x) @ self.WO.T) * self.OUT_GAIN @ E.T

    def constants(self) -> dict:
        with torch.no_grad():
            c = {k: getattr(self, k).detach().numpy().copy() for k in ("POS", "WQ", "WK", "WV", "WO", "OUT_GAIN")}
            c["EMBED"] = self.embed().detach().numpy().copy()
        c["LAYERNORM"] = self.ln
        return c


def fit(model, step: float, seed: int, steps: int = 3000, lr: float = 3e-3, bs: int = 64) -> dict:
    torch.manual_seed(seed)
    acc = Account(constants(model), step)
    sampler = BatchSampler(data_char.load_stream("train"), 2000 + seed, bs, model.cfg.context_length)
    opt = torch.optim.Adam(acc.parameters(), lr=lr)
    for i in range(steps):
        x = torch.as_tensor(sampler.batch(i))[:, :-1]
        with torch.no_grad():
            lp_m = torch.log_softmax(model(x), -1)
        lp_a = torch.log_softmax(acc(x), -1)
        loss = (lp_m.exp() * (lp_m - lp_a)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * i / steps)))
    return acc.constants()


def score(model, c: dict, W: np.ndarray) -> float:
    acc = Account({**c, **{k: np.round(v, 2) for k, v in c.items() if k not in ("EMBED", "LAYERNORM")}}, 0)
    with torch.no_grad():
        lp = np.concatenate([torch.log_softmax(acc(torch.as_tensor(W[i:i + 200, :-1].astype(np.int64))), -1).numpy()
                             for i in range(0, len(W), 200)])
    return float(kl_pos(model_logprobs(model, W), lp).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model = load("L1", a.arm, a.seed).double()
    c = fit(model, a.step, a.seed, a.steps)
    print(f"seed {a.seed} step {a.step}: KL on dev_full[:1000] {score(model, c, windows('L1', 'dev_full')[:1000]):.4f}")


if __name__ == "__main__":
    main()
