"""Fidelity to what all seeds share (analysis only; TRAIN windows).

ENS = probability average of attn_s0..s4. For each model and for the program: KL(ENS || x); and leave-one-out:
KL(seed k || average of the other four). The program is fitted to attn_s0 only.

    .venv/bin/python -m analysis.program_gap.ensemble_target [--n 64]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.program_gap.bottleneck import OUT, kl_rows
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model, model_logprobs

V = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    W = train_windows(args.n, seed=EVAL_SEED)
    seeds = [f"attn_s{k}" for k in range(5)]
    lps = {m: model_logprobs(load_model(m), W).reshape(-1, V) for m in seeds}
    P = program()
    lq = np.concatenate([P.predict(np.asarray(w[:256], np.int64)) for w in W])
    ens = np.log(np.mean([np.exp(lps[m]) for m in seeds], 0))
    res = {"n_windows": args.n, "KL(ENS5 || x)": {m: float(kl_rows(ens, lps[m]).mean()) for m in seeds}}
    res["KL(ENS5 || x)"]["program"] = float(kl_rows(ens, lq).mean())
    res["KL(seed k || mean of other 4)"] = {}
    for m in seeds:
        o = np.log(np.mean([np.exp(lps[x]) for x in seeds if x != m], 0))
        res["KL(seed k || mean of other 4)"][m] = float(kl_rows(lps[m], o).mean())
    res["KL(x || ENS5)"] = {m: float(kl_rows(lps[m], ens).mean()) for m in seeds}
    res["KL(x || ENS5)"]["program"] = float(kl_rows(lq, ens).mean())
    res["KL(seed k || program)"] = {m: float(kl_rows(lps[m], lq).mean()) for m in seeds}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "ensemble_target.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
