"""EXPLORATORY (logs/prereg_fixed_votes.md, "Exploratory": a readable description and its KL to the model). In a
fixed-vote model the votes are named corpus tables and the strengths are five numbers, so replacing every head's
attention with a readable rule gives a fully readable program. This script fits, per head, goal 1's base rule
(distance buckets, self x query shape, key shape) with and without the head's own-usefulness features (corpus
statistics) on train_windows(32, 2024), substitutes ALL four heads' rule patterns at once on train_windows(32, 2025),
and reports KL(model || program), KL(attn_s0 || program), the cross-entropy on the next tokens, and the same for the
fixed-vote model itself and for the non-readable token-token ceiling (reference).

    OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.fixed_votes.readable_all models/fixed_votes/base_s0/final.pt

Writes analysis/out/fixed_votes/<name>/readable_all.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from analysis.fixed_votes import law as FL
from analysis.look_write import common as C
from analysis.look_write.fit_arms import Rule, fit_rule, tt_feature

ARMS = ("base", "own", "tt")          # tt: the exact token-token term, not readable (reference ceiling)


def logprobs_all(m, wins, pats=None, bs=8):
    """Log-probs (n, T, V) with every head h's pattern replaced by pats[h] (n, T, T) when pats is given."""
    out = []
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs], np.int64))
            ov = None if pats is None else {h: torch.as_tensor(p[i:i + bs]) for h, p in pats.items()}
            out.append(torch.log_softmax((m(x) if ov is None else m(x, override=ov)).double(), -1).float().numpy())
    return np.concatenate(out)


def ce_next(lp, wins):
    """Mean cross-entropy of the next tokens (positions 0..T-2 predict 1..T-1)."""
    w = np.asarray(wins, np.int64)
    return float(-np.take_along_axis(lp[:, :-1], w[:, 1:, None], -1).mean())


def run(path):
    from src.train.train import load_model

    C.set_threads()
    m = load_model(path)
    for p in m.parameters():
        p.requires_grad_(False)
    names, beta = m.cfg.names, [float(b) for b in m.beta]
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    A_fit = FL.patterns(m, fit_w)
    rf, re = Rule(fit_w, shapes), Rule(ev_w, shapes)
    tt = C.tt_tables(FL.fv_weights(m))
    pats = {a: {} for a in ARMS}
    for h in range(m.cfg.n_heads):
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        feats = {"base": ({}, {}),
                 "own": (FL.u_features(fit_w, names[h], beta[h]), FL.u_features(ev_w, names[h], beta[h])),
                 "tt": ({"c": torch.as_tensor(tt_feature(fit_w, tt[h]))}, {"c": torch.as_tensor(tt_feature(ev_w, tt[h]))})}
        for a, (ff, fe) in feats.items():
            theta, _, _ = fit_rule(A_f, rf, ff)
            with torch.no_grad():
                pats[a][h] = torch.exp(re.logpattern(theta, fe)).numpy()
        print(f"head {h} ({names[h]}) fitted", flush=True)
    lp_model = logprobs_all(m, ev_w)
    lp_attn = logprobs_all(load_model(os.path.join(C.REPO, "models", "final", "attn_s0", "final.pt")), ev_w)
    res = {"model": path, "windows": "train_windows(32, 2025)", "fixed_vote_model": {
        "ce": ce_next(lp_model, ev_w), "kl_attn_s0_to_this": C.kl(lp_attn, lp_model)},
        "attn_s0_ce": ce_next(lp_attn, ev_w), "programs": {}}
    for a in ARMS:
        lp = logprobs_all(m, ev_w, pats[a])
        res["programs"][a] = {"readable": a != "tt", "kl_model_to_program": C.kl(lp_model, lp),
                              "kl_attn_s0_to_program": C.kl(lp_attn, lp), "ce": ce_next(lp, ev_w)}
    d = os.path.join(FL.OUT, FL.name_of(path))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "readable_all.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
