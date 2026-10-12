"""Fit the readable program to a model and write it to program/tables/.

  python -m analysis.run_fit --model attn_s0 --n_follow 20000 --n_recall 5000 --tag v1

Stages:
  0. Initialise WORD from the model's mean prediction (analysis/mean_dists.py output if present).
  1. Fit each head's attention tables to the matching model head's attention pattern.
  2. Fit the class-level tables with exceptions held at zero, then add dense exceptions on top.
  3. Keep the --n_follow / --n_recall most important exceptions (|value| * sqrt(token frequency)),
     zero and freeze the rest, and refit.
  4. Report KL on held-out TRAIN windows, write tables, check program == student numerically.
Student checkpoints go to analysis/out/<model>/student_<tag>.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from analysis.fitlib import REPO, Target, model_name, train_windows
from analysis.student import Student, evaluate, fit, fit_attention, prune_to_budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n_windows", type=int, default=8000)
    ap.add_argument("--attn_steps", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--refit_steps", type=int, default=1500)
    ap.add_argument("--class_steps", type=int, default=1500, help="stage 2a: class-level only")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--n_follow", type=int, default=20000, help="AFTER exceptions kept")
    ap.add_argument("--n_recall", type=int, default=5000, help="RECALL exceptions kept per head")
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--no_emit", action="store_true")
    ap.add_argument("--init", default=None, help="student checkpoint to start from")
    ap.add_argument("--freeze", default="", help="comma list of Student parameters held at zero, e.g. class_follow")
    args = ap.parse_args()

    from program.tables_lib import TOKEN_CLASS

    mname = model_name(args.model)
    out = os.path.join(REPO, "analysis", "out", mname)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tgt = Target(args.model, args.device)
    W = torch.as_tensor(train_windows(args.n_windows))
    W_hold = torch.as_tensor(train_windows(200, seed=2000))
    st = Student(args.n_heads, TOKEN_CLASS, int(TOKEN_CLASS.max()) + 1)
    md = os.path.join(out, "mean_dists.npz")
    if args.init:
        st.load_state_dict(torch.load(args.init))
    elif os.path.exists(md):
        uni = np.load(md)["uni"]
        with torch.no_grad():
            st.word.copy_(torch.as_tensor(np.log(uni + 1e-9), dtype=torch.float32))
    with torch.no_grad():  # break symmetry between heads before stage 1
        st.recency.normal_(0, 0.5)
    st.to(args.device)

    log = {"model": args.model, "tag": args.tag, "args": vars(args)}
    freeze = tuple(f for f in args.freeze.split(",") if f)
    if not args.init and args.attn_steps:
        log["pattern_kl"] = fit_attention(st, tgt, W, steps=args.attn_steps, bs=args.bs)
    if args.class_steps:  # stage 2a: class-level tables only (exceptions held at zero)
        fm, rm = st.follow_mask.clone(), st.recall_mask.clone()
        st.follow_mask.zero_()
        st.recall_mask.zero_()
        fit(st, args.model, steps=args.class_steps, bs=args.bs, lr=args.lr, device=args.device, tgt=tgt,
            W=W, log_every=200, seed=3, freeze=freeze)
        log["kl_class_level"] = evaluate(st, tgt, W_hold)
        st.follow_mask.copy_(fm)
        st.recall_mask.copy_(rm)
    fit(st, args.model, steps=args.steps, bs=args.bs, lr=args.lr, device=args.device, tgt=tgt, W=W,
        tau=args.tau, log_every=200, freeze=freeze)
    log["kl_after_dense_fit"] = evaluate(st, tgt, W_hold)
    freq = np.load(md)["count"] if os.path.exists(md) else np.ones(2000)
    log["pruned"] = prune_to_budget(st, args.n_follow, args.n_recall, freq + 1.0)
    log["kl_after_prune"] = evaluate(st, tgt, W_hold)
    fit(st, args.model, steps=args.refit_steps, bs=args.bs, lr=args.lr * 0.3, device=args.device, tgt=tgt,
        W=W, tau=0.0, log_every=200, seed=5, freeze=freeze)
    log["kl_holdout_train"] = evaluate(st, tgt, W_hold)
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=1))
    st.cpu()
    ck = os.path.join(out, f"student_{args.tag}.pt")
    torch.save(st.state_dict(), ck)
    json.dump(log, open(os.path.join(out, f"fit_{args.tag}.json"), "w"), indent=1)
    if args.no_emit:
        return
    from analysis.emit_program import emit

    emit(st, mname, f"Fit tag '{args.tag}'.")
    r = subprocess.run([sys.executable, "-m", "analysis.check_emit", ck], cwd=REPO, capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:])


if __name__ == "__main__":
    main()
