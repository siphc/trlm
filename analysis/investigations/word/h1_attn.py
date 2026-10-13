"""Does head 1 look for earlier word pieces that can follow the current token (token-level successor), or for the
token that followed an earlier copy of the current token (induction)? Fitted to head 1's OWN attention rows.

Rule: head 1's program score (program/head1.py, frozen) + B_SUCC * succ(t_i, t_j) + B_IND * [t_{j-1} = t_i, j < i]
  succ(q, k) = max(0, PMI_1(q, k)), PMI_1 the corpus bigram PMI (corpus.py), token level.
Pattern KL(head || rule) per query, by query class; 32 TRAIN windows (fit seed 2024), evaluated on 32 others.

  python -m analysis.investigations.word.h1_attn
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.decomp import parts
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model
from program import head1

FIT_SEED = 2024


def rule_parts(P, t):
    s = P.STATS
    n = len(t)
    dist = np.arange(n)[:, None] - np.arange(n)[None, :]
    base = (-dist / head1.MEMORY + head1._per_group(head1.SALIENCE)[head1.GROUP[t]][None, :]
            + head1.SELF_BONUS * np.eye(n) + head1.SUCCESSOR * s.class_successor[t[:, None], t[None, :]])
    succ = np.maximum(np.asarray(s.pmi[1])[t[:, None], t[None, :]], 0.0) * (dist >= 1)
    ind = np.zeros((n, n))
    prev = np.r_[-1, t[:-1]]
    ind[:, :] = (prev[None, :] == t[:, None]) & (dist >= 1)
    return base, succ, ind


def collect(P, model, W, PIECE):
    out = []
    with torch.no_grad():
        for w in W:
            t = np.asarray(w[:256], dtype=np.int64)
            a, _, _, _ = parts(model, torch.as_tensor(t[None]))
            A = a[0, 1].double().numpy()
            base, succ, ind = rule_parts(P, t)
            cl, _ = classes(P, t, PIECE)
            cl = np.array(["inside" if c.startswith("inside") else c for c in cl], dtype=object)
            out.append((A, base, succ, ind, cl))
    return out


def pattern_kl(data, b_succ, b_ind):
    res = {}
    for A, base, succ, ind, cl in data:
        n = len(cl)
        score = base + b_succ * succ + b_ind * ind
        score = np.where(np.tril(np.ones((n, n), bool)), score, -np.inf)
        logr = score - score.max(1, keepdims=True)
        logr = logr - np.log(np.exp(logr).sum(1, keepdims=True))
        kl = np.where(A > 0, A * (np.log(np.maximum(A, 1e-300)) - np.where(np.isfinite(logr), logr, 0)), 0).sum(1)
        for c in set(cl):
            res.setdefault(c, []).append(kl[cl == c])
        res.setdefault("all", []).append(kl)
    return {c: float(np.concatenate(v).mean()) for c, v in res.items()}


def fit(data, use_succ=True, use_ind=True):
    b = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    tens = [(torch.as_tensor(A), torch.as_tensor(base), torch.as_tensor(succ), torch.as_tensor(ind))
            for A, base, succ, ind, _ in data]
    mask = torch.as_tensor(np.tril(np.ones((256, 256), bool)))
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for A, base, succ, ind in tens:
            sc = base + (b[0] * succ if use_succ else 0) + (b[1] * ind if use_ind else 0)
            sc = sc.masked_fill(~mask, float("-inf"))
            loss = loss - (A * torch.log_softmax(sc, 1).masked_fill(~mask, 0)).sum() / (256 * len(tens))
        loss.backward()
        return loss

    opt.step(closure)
    return [float(x) for x in b.detach()]


def main():
    P = program()
    PIECE = WS.piece_flags()
    model = load_model("attn_s0")
    fitd = collect(P, model, train_windows(32, seed=FIT_SEED), PIECE)
    evald = collect(P, model, train_windows(32, seed=EVAL_SEED), PIECE)
    out = {"program rule": {"params": [0, 0], "pattern_kl": pattern_kl(evald, 0, 0)}}
    for label, us, ui in (("+ successor", True, False), ("+ induction", False, True), ("+ both", True, True)):
        b = fit(fitd, us, ui)
        b = [b[0] if us else 0.0, b[1] if ui else 0.0]
        out[label] = {"params": [round(x, 3) for x in b], "pattern_kl": pattern_kl(evald, *b)}
    for k, v in out.items():
        print(f"{k:14s} B_SUCC {v['params'][0]:+.3f} B_IND {v['params'][1]:+.3f}  " +
              "  ".join(f"{c} {x:.3f}" for c, x in sorted(v["pattern_kl"].items())), flush=True)
    with open(os.path.join(WS.OUT, "h1_attn.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
