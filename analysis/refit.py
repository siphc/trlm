"""Refit a pruned Student with its zero pattern fixed, optionally freezing named parameters.

Used for paired structural tests at a fixed budget: run once with a feature frozen at zero (control) and once
with it free; both get the same steps, data and seed, so the difference is the feature's worth.

  python -m analysis.refit --init analysis/out/attn_s0/student_<tag>.pt --steps 1000 --freeze log_conf --tag ctl
  python -m analysis.refit --init analysis/out/attn_s0/student_<tag>.pt --steps 1000 --tag conf --save
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import REPO, Target, ensure_disk, model_name, train_windows
from analysis.student import Student, evaluate, fit, load_student, save_student


@torch.no_grad()
def per_window_kl(st: Student, tgt: Target, W: torch.Tensor, bs: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(W), bs):
        x = W[i : i + bs, :256].to(tgt.device)
        p_log = tgt.logprobs(x)
        out.append((p_log.exp() * (p_log - st(x))).sum(-1).mean(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--freeze", default="", help="comma list of Student parameter names kept fixed")
    ap.add_argument("--n_windows", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--confidence", action="store_true", help="also fit CONFIDENCE")
    args = ap.parse_args()

    st = load_student(args.init)
    st.to(args.device)
    tgt = Target(args.model, args.device)
    W = torch.as_tensor(train_windows(args.n_windows))
    W_hold = torch.as_tensor(train_windows(200, seed=2000))
    k0 = per_window_kl(st, tgt, W_hold)
    freeze = tuple(f for f in args.freeze.split(",") if f)
    fit(st, args.model, steps=args.steps, bs=args.bs, lr=args.lr, device=args.device, tgt=tgt, W=W,
        log_every=10**9, seed=args.seed, freeze=freeze, train_conf="log_conf" not in freeze and args.confidence)
    k1 = per_window_kl(st, tgt, W_hold)
    out = os.path.join(REPO, "analysis", "out", model_name(args.model))
    res = {"init": args.init, "tag": args.tag, "freeze": freeze, "steps": args.steps,
           "kl_before": float(k0.mean()), "kl_after": float(k1.mean())}
    np.save(os.path.join(out, f"refit_{args.tag}_perwindow.npy"), k1)
    json.dump(res, open(os.path.join(out, f"refit_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(res))
    if args.save:
        ensure_disk(1.0)
        save_student(st.cpu(), os.path.join(out, f"student_{args.tag}.pt"))


if __name__ == "__main__":
    main()
