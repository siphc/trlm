"""Fit every named scalar of the mechanism program to one model, alternating until attention and votes agree.

  .venv/bin/python scripts/fit_scalars.py [--model attn_s0] [--assign head0=0,head2=2,head1=1] [--iterations 3]
                                          [--write program]

Why alternate: the local readers' attention includes the usefulness of their OWN vote (readers.vote_usefulness),
which depends on their vote strengths, and the best vote strengths depend on the attention. Each round:
  1. attention: each head rule (program/head0.py, head2.py, head1.py) is fitted to its assigned model head's own
     attention pattern (pattern cross-entropy, train_windows(32, 2024); analysis/attention_fit.py);
  2. votes: all vote strengths are refitted jointly to the model's log-probs (analysis/vote_fit.py, same fit
     windows; convex, except RECALL's vote size when the program has one, fitted with them).
Reported each round: held-out pattern KL per head (train_windows(16, 2025)) and held-out output KL
(train_windows(64, 2000)). With --write, the final constants are written into that program directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def set_strengths(strengths):
    from mech_seed_transfer import strengths_to_constants
    from program import follow, head0, head1, head2, phrase

    mods = {"follow": follow, "head0": head0, "head1": head1, "head2": head2, "phrase": phrase}
    consts = strengths_to_constants(strengths)
    for m, c in consts.items():
        for name, value in c.items():
            setattr(mods[m], name, value)
    return consts


def attention_constants(mod, fitted):
    out = {}
    for key, v in fitted.items():
        attr, _, idx = key.partition("[")
        if idx:
            cur = out.setdefault(attr, getattr(mod, attr))
            cur = list(cur) if isinstance(cur, (list, tuple)) else dict(cur)
            k = idx[:-1]
            cur[int(k) if k.isdigit() else k.strip("'\"")] = v
            out[attr] = cur
        else:
            out[attr] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--assign", default="head0=0,head2=2,head1=1")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--write", default="")
    ap.add_argument("--out_dir", default="", help="where fit_scalars.json goes (default analysis/out/<model>)")
    args = ap.parse_args()
    if args.write.rstrip("/") == "program" and args.model != "attn_s0":
        raise SystemExit("refusing to write another model's constants into program/ (use a copy)")
    import torch

    from analysis import vote_fit
    from analysis.attention_fit import fit_rule, model_attention, pattern_ce
    from analysis.fitlib import train_windows
    from analysis.mech_eval import program
    from harness.common import load_model, model_logprobs
    from program import head0, head1, head2
    from program.tests import measure as M

    torch.set_num_threads(2)
    mods = {"head0": head0, "head1": head1, "head2": head2}
    assign = {k: int(v) for k, v in (x.split("=") for x in args.assign.split(","))}
    P = program()
    s = P.STATS
    Wf, Wh, We = train_windows(32, seed=2024), train_windows(16, seed=2025), train_windows(64, seed=2000)
    Af, Ah = model_attention(args.model, Wf), model_attention(args.model, Wh)
    model = load_model(args.model)
    probs = np.exp(model_logprobs(model, Wf)).reshape(-1, 2000).astype(np.float32)
    lpe = model_logprobs(model, We)

    def heldout():
        kl = float(np.mean([M.kl(p, P.predict(np.asarray(w[:256], np.int64))).mean() for w, p in zip(We, lpe)]))
        pat = {r: float(pattern_ce(mods[r], Ah[:, h], Wh, s)[1]) for r, h in assign.items()}
        return kl, pat

    history = [{"round": 0, **dict(zip(("output_kl", "pattern_kl"), heldout()))}]
    print(json.dumps(history[-1]), flush=True)
    consts = {}
    for it in range(1, args.iterations + 1):
        attn = {}
        for role, h in assign.items():
            fitted = fit_rule(mods[role], Af[:, h], Wf, s, keep=True)
            attn[role] = attention_constants(mods[role], fitted)
        X, keys, b0 = vote_fit.stacked(P, Wf, None)
        sat0 = vote_fit.saturation_of_program()
        if sat0 is None:
            beta = vote_fit.fit(X, probs, [b0[k] for k in keys])
        else:
            beta, sat = vote_fit.fit(X, probs, [b0[k] for k in keys], keys=keys, c0=sat0, pu=s.p_uni)
        del X
        consts = set_strengths(dict(zip(keys, map(float, beta))))
        if sat0 is not None:
            head1.SIZE = sat
            consts["head1"]["SIZE"] = sat
        for role, c in attn.items():
            consts[role].update(c)
        kl, pat = heldout()
        history.append({"round": it, "output_kl": kl, "pattern_kl": pat})
        print(json.dumps(history[-1]), flush=True)
        if abs(history[-2]["output_kl"] - kl) < 0.0003:
            break
    out_dir = args.out_dir or os.path.join(REPO, "analysis", "out", args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fit_scalars.json"), "w") as f:
        json.dump({"model": args.model, "assign": assign, "history": history, "constants": consts}, f, indent=1,
                  default=float)
    if args.write:
        from mech_seed_transfer import rewrite_constants

        rewrite_constants(os.path.join(REPO, args.write), consts,
                          f"fitted to {args.model} (scripts/fit_scalars.py)")
        print("constants written to", args.write)


if __name__ == "__main__":
    main()
