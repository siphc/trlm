"""Gradual pruning of a dense Student down to an exception budget, refitting after every round.

One-shot pruning of 99.8% of the exceptions is far outside the regime where per-entry importance is
meaningful (logs/dead_ends.md). Here the budget shrinks geometrically over --rounds rounds; after each cut
the surviving tables are refitted with the zero pattern frozen, so information migrates into the kept
entries before the next cut. Every round's held-out KL is logged, which traces fidelity vs size.

  python -m analysis.prune --init analysis/out/attn_s0/student_dense_ref_dense.pt --model attn_s0 \
      --target 60000 --rounds 8 --refit_steps 300 --tag gp60k

Ranking (--rank): "data" = first-order data importance with the full program (student.data_importance);
"fisher" = 0.5 w^2 F (second order, student.fisher_importance); "magnitude" = |w|. FOLLOW and RECALL compete for one global budget unless --follow_share is given.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.fitlib import REPO, Target, model_name, train_windows
from analysis.student import Student, data_importance, evaluate, fisher_importance, fit, make_nonredundant


def apply_budget(st: Student, n_keep: int, W, rank: str, follow_share: float | None,
                 importance_batches: int = 40) -> dict:
    with torch.no_grad():
        if rank == "data":
            i_f, i_r = data_importance(st, W, n_batches=importance_batches)
        elif rank == "fisher":
            i_f, i_r = fisher_importance(st, W, n_batches=importance_batches)
        else:
            i_f = (st.follow_exc * st.follow_mask).abs()
            i_r = (st.recall_exc * st.recall_mask).abs()

        def keep_top(imp, n):
            flat = imp.flatten()
            n_pos = int((flat > 0).sum())
            if n >= n_pos:
                return (imp > 0).float()
            thr = torch.topk(flat, n).values[-1]
            return ((imp >= thr) & (imp > 0)).float()

        if follow_share is None:
            allimp = torch.cat([i_f.flatten(), i_r.flatten()])
            n_pos = int((allimp > 0).sum())
            if n_keep < n_pos:
                thr = torch.topk(allimp, n_keep).values[-1]
                st.follow_mask *= ((i_f >= thr) & (i_f > 0)).float()
                st.recall_mask *= ((i_r >= thr) & (i_r > 0)).float()
        else:
            nf = int(round(n_keep * follow_share))
            st.follow_mask *= keep_top(i_f, nf)
            st.recall_mask *= keep_top(i_r, n_keep - nf)
        st.follow_exc *= st.follow_mask
        st.recall_exc *= st.recall_mask
    return {"follow": int(st.follow_mask.sum()), "recall": [int(st.recall_mask[h].sum()) for h in range(st.H)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="dense Student checkpoint")
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", type=int, required=True, help="final number of exceptions (all tables)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--refit_steps", type=int, default=300)
    ap.add_argument("--final_refit_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--rank", default="data", choices=["data", "fisher", "magnitude"])
    ap.add_argument("--nonredundant", action="store_true",
                    help="first move heads' current-token votes into FOLLOW and silence heads on self")
    ap.add_argument("--freeze_attention", action="store_true", help="refits keep attention tables fixed")
    ap.add_argument("--follow_share", type=float, default=None)
    ap.add_argument("--n_windows", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--save_rounds", action="store_true", help="save the student after every round")
    ap.add_argument("--importance_batches", type=int, default=40,
                    help="batches of 16 windows used to estimate data importance (rare tokens need many)")
    args = ap.parse_args()

    from program.tables_lib import TOKEN_CLASS

    sd = torch.load(args.init, map_location="cpu")
    st = Student(sd["salience"].shape[0], TOKEN_CLASS, int(TOKEN_CLASS.max()) + 1)
    st.load_state_dict(sd)
    st.to(args.device)
    tgt = Target(args.model, args.device)
    W = torch.as_tensor(train_windows(args.n_windows))
    W_hold = torch.as_tensor(train_windows(200, seed=2000))
    if args.nonredundant:
        make_nonredundant(st, W)
        print(f"non-redundant transform: holdout KL {evaluate(st, tgt, W_hold):.4f}", flush=True)
    n0 = int(st.follow_mask.sum() + st.recall_mask.sum())
    budgets = np.geomspace(n0, args.target, args.rounds + 1)[1:].round().astype(int)
    log = {"init": args.init, "target": args.target, "rank": args.rank, "rounds": [],
           "kl_start": evaluate(st, tgt, W_hold)}
    print(f"start: {n0:,} exceptions, holdout KL {log['kl_start']:.4f}", flush=True)
    t0 = time.time()
    for r, b in enumerate(budgets):
        kept = apply_budget(st, int(b), W, args.rank, args.follow_share, args.importance_batches)
        kl_cut = evaluate(st, tgt, W_hold)
        steps = args.final_refit_steps if r == len(budgets) - 1 else args.refit_steps
        fit(st, args.model, steps=steps, bs=args.bs, lr=args.lr, device=args.device, tgt=tgt, W=W,
            log_every=10**9, seed=100 + r, freeze_attention=args.freeze_attention)
        kl_refit = evaluate(st, tgt, W_hold)
        row = {"round": r, "budget": int(b), "kept": kept, "kl_after_cut": kl_cut, "kl_after_refit": kl_refit,
               "t": time.time() - t0}
        log["rounds"].append(row)
        if args.save_rounds:
            torch.save({k: v.detach().cpu() for k, v in st.state_dict().items()},
                       os.path.join(REPO, "analysis", "out", model_name(args.model), f"student_{args.tag}_r{r}.pt"))
        print(f"round {r}: budget {b:,}  kept follow {kept['follow']:,} recall {kept['recall']}  "
              f"KL cut {kl_cut:.4f} -> refit {kl_refit:.4f}  ({row['t']:.0f}s)", flush=True)
    out = os.path.join(REPO, "analysis", "out", model_name(args.model))
    st.cpu()
    torch.save(st.state_dict(), os.path.join(out, f"student_{args.tag}.pt"))
    json.dump(log, open(os.path.join(out, f"prune_{args.tag}.json"), "w"), indent=1)
    if args.emit:
        from analysis.emit_program import emit

        emit(st, model_name(args.model), f"Gradual pruning tag '{args.tag}'.")


if __name__ == "__main__":
    main()
