"""Shared CPU evaluation for design experiments (keeps numbers comparable across agents).

    from analysis.scratch_eval import load_dense, holdout, kl_of
    st = load_dense()                       # dense reference Student (all exceptions dense), CPU
    W = holdout(48)                         # fixed held-out TRAIN windows (seed 2000), first n
    print(kl_of(st, W))                     # mean per-token KL(model || student) on those windows

Single-threaded by default (torch.set_num_threads(1)) so experiments do not starve the training run.
The reference numbers on holdout(48) are printed by `python -m analysis.scratch_eval`.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from analysis.fitlib import REPO, Target, train_windows
from analysis.student import Student, evaluate

torch.set_num_threads(int(os.environ.get("SCRATCH_THREADS", "1")))
DENSE = os.path.join(REPO, "analysis", "out", "attn_s0", "student_dense_ref_dense.pt")
_TGT = None


def target() -> Target:
    global _TGT
    if _TGT is None:
        _TGT = Target("attn_s0", "cpu")
    return _TGT


def load_dense(path: str = DENSE) -> Student:
    from program.tables_lib import TOKEN_CLASS

    sd = torch.load(path, map_location="cpu")
    st = Student(sd["salience"].shape[0], TOKEN_CLASS, int(TOKEN_CLASS.max()) + 1)
    st.load_state_dict(sd)
    return st.eval()


def holdout(n: int = 48) -> torch.Tensor:
    return torch.as_tensor(train_windows(200, seed=2000))[:n]


def kl_of(st: Student, W: torch.Tensor, bs: int = 4) -> float:
    return evaluate(st, target(), W, bs=bs)


if __name__ == "__main__":
    import time

    t0 = time.time()
    st = load_dense()
    print(f"dense reference on holdout(48): KL {kl_of(st, holdout(48)):.4f}  ({time.time() - t0:.0f}s)")
