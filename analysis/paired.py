"""Paired comparison of students on held-out TRAIN windows (the adoption protocol).

Per-window KL(model || student) on train_windows(200, seed=2000) is cached per checkpoint in
analysis/out/<model>/pw_<checkpoint name>.npy. The window is the sampling unit (positions inside a window are
correlated), so the standard error of a difference is sd(per-window difference) / sqrt(200).

  python -m analysis.paired analysis/out/attn_s0/student_A.pt analysis/out/attn_s0/student_B.pt

Adoption rule for fidelity changes (design-panel synthesis): both arms at the same budget and pipeline;
paired gain >= 0.005 and >= 3 SE (>= 0.01 if the change adds a class system or a program component).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from analysis.fitlib import REPO, Target, train_windows
from analysis.student import load_student


@torch.no_grad()
def per_window(ckpt: str, model: str = "attn_s0", device: str = "cpu", bs: int = 8) -> np.ndarray:
    cache = os.path.join(os.path.dirname(ckpt), "pw_" + os.path.basename(ckpt).replace(".pt", ".npy"))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(ckpt):
        return np.load(cache)
    st = load_student(ckpt).to(device).eval()
    tgt = Target(model, device)
    W = torch.as_tensor(train_windows(200, seed=2000))
    out = []
    for i in range(0, len(W), bs):
        x = W[i : i + bs, :256].to(device)
        p_log = tgt.logprobs(x)
        out.append((p_log.exp() * (p_log - st(x))).sum(-1).mean(-1).cpu().numpy())
    pw = np.concatenate(out)
    np.save(cache, pw)
    return pw


def compare(a: str, b: str) -> dict:
    A, B = per_window(a), per_window(b)
    d = A - B
    res = {"a": a, "b": b, "mean_a": float(A.mean()), "mean_b": float(B.mean()), "diff": float(d.mean()),
           "se": float(d.std(ddof=1) / np.sqrt(len(d))), "frac_a_better": float((d < 0).mean())}
    print(f"{os.path.basename(a)} - {os.path.basename(b)}: {res['mean_a']:.4f} - {res['mean_b']:.4f} = "
          f"{res['diff']:+.4f} +- {res['se']:.4f} (A better on {res['frac_a_better']:.0%} of windows)")
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("PAIRED_THREADS", "4")))
    args = sys.argv[1:]
    for a, b in zip(args[::2], args[1::2]):
        compare(a, b)
