"""Does model head 1's contribution saturate where RECALL's raw vote is large? (goal 2, analysis only)

Per position (TRAIN windows): size of model head 1's logit contribution (current + earlier part, final LN at the
actual scale; unigram-weighted standard deviation over the vocabulary) against the size of the program's raw
(unsaturated) RECALL vote, and against head 1's attention concentration (sum of squared attention weights).

    .venv/bin/python -m analysis.goal2.saturation_check [--n 16]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.attention_fit import model_attention
from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT
from analysis.goal2.budget import model_parts
from analysis.mech_eval import EVAL_SEED, program


def wstd(x, w):
    mu = x @ w
    return np.sqrt(((x - mu[:, None]) ** 2) @ w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    from harness.common import load_model
    from program import head1

    torch.set_num_threads(3)
    model = load_model("attn_s0")
    P = program()
    s = P.STATS
    w = s.p_uni
    W = train_windows(args.n, seed=EVAL_SEED)
    A = model_attention(model, W)[:, 1]                                 # (N, T, T) head 1
    ms, ps, conc, pos = [], [], [], []
    for k, win in enumerate(W):
        t = np.asarray(win[:256], np.int64)
        mp, _ = model_parts(model, t)
        ms.append(wstd(mp["self1"] + mp["earlier1"], w))
        raw = head1.attention(t, s) @ head1.recalled_votes(t, s)
        ps.append(wstd(raw, w))
        conc.append((A[k].astype(np.float64) ** 2).sum(-1))
        pos.append(np.arange(256))
    ms, ps, conc, pos = map(np.concatenate, (ms, ps, conc, pos))
    res = {"n_windows": args.n, "corr(model size, program raw size)": float(np.corrcoef(ms, ps)[0, 1]),
           "corr(model size, head-1 attention concentration)": float(np.corrcoef(ms, conc)[0, 1])}
    edges = np.quantile(ps, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, ps, side="right") - 1, 0, 9)
    res["by program raw size decile"] = [{"program raw size": float(ps[b == i].mean()), "model size": float(ms[b == i].mean())}
                                         for i in range(10)]
    edges = np.quantile(conc, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, conc, side="right") - 1, 0, 9)
    res["by model attention concentration decile"] = [{"concentration": float(conc[b == i].mean()),
                                                        "model size": float(ms[b == i].mean())} for i in range(10)]
    res["position 0"] = {"model size": float(ms[pos == 0].mean()), "program raw size": float(ps[pos == 0].mean())}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "saturation_check.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
