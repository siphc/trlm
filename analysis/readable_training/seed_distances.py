"""Tool e: distances between models on train_windows(64, 9100): pairwise KL, KL to the five-seed ensemble of
attn_s0..s4 (probability average), KL to program v1.3, and attn_s0 specificity
(KL(m || attn_s0) - mean_k KL(m || attn_sk), k = 1..4).

    .venv/bin/python -m analysis.readable_training.seed_distances <name=model or path> ...
Writes analysis/out/readable_training/seed_distances.json (merged with earlier runs).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from analysis.readable_training.common import OUT

BASE = [f"attn_s{k}" for k in range(5)]


def main(specs):
    from analysis.fitlib import train_windows
    from analysis.readable_training import evalsets as E
    from harness.common import load_model, model_logprobs

    W = train_windows(64, 9100)
    models = dict(s.split("=", 1) if "=" in s else (s, s) for s in specs)
    for b in BASE:
        models.setdefault(b, b)
    LP = {n: model_logprobs(load_model(p), W).astype(np.float32) for n, p in models.items()}
    prog = np.asarray(E.program_lp("train_eval"), np.float32)
    prog = prog - np.log(np.exp(prog).sum(-1, keepdims=True))
    ens = np.log(np.mean([np.exp(LP[b]) for b in BASE], 0))
    kl = lambda a, b: float((np.exp(a) * (a - b)).sum(-1).mean())   # noqa: E731
    names = list(models)
    res = {"pairwise_kl": {a: {b: kl(LP[a], LP[b]) for b in names if b != a} for a in names},
           "kl_to_ensemble5": {n: kl(LP[n], ens) for n in names},
           "kl_to_program": {n: kl(LP[n], prog) for n in names},
           "attn_s0_specificity": {n: kl(LP[n], LP["attn_s0"]) - np.mean([kl(LP[n], LP[f"attn_s{k}"])
                                                                         for k in range(1, 5)]) for n in names}}
    p = os.path.join(OUT, "seed_distances.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    for k, v in res.items():
        old.setdefault(k, {}).update(v)
    os.makedirs(OUT, exist_ok=True)
    with open(p, "w") as f:
        json.dump(old, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pairwise_kl"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
