"""How far apart are models that differ only in seed? (goal 2; the floor for any seed-agnostic explanation)

KL(attn_s0 || attn_sK) and KL(attn_s0 || mixture-of-other-seeds) on TRAIN windows, next to the program's KL to
attn_s0; also the current-token tables' agreement across seeds (same construction as table_probe.py).

    .venv/bin/python -m analysis.goal2.seed_floor [--n 64]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT, kl_rows
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model, model_logprobs

V = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    W = train_windows(args.n, seed=EVAL_SEED)
    lps = {m: model_logprobs(load_model(m), W).reshape(-1, V) for m in [f"attn_s{k}" for k in range(5)] + ["mlp_s0"]}
    P = program()
    lq = np.concatenate([P.predict(np.asarray(w[:256], np.int64)) for w in W])
    ref = lps["attn_s0"]
    res = {"n_windows": args.n, "program (v1.2) vs attn_s0": float(kl_rows(ref, lq).mean())}
    for m, lp in lps.items():
        if m != "attn_s0":
            res[f"{m} vs attn_s0"] = float(kl_rows(ref, lp).mean())
    others = [lps[f"attn_s{k}"] for k in range(1, 5)]
    mix = np.log(np.mean([np.exp(o) for o in others], 0))
    res["mean of seeds 1-4 (probability average) vs attn_s0"] = float(kl_rows(ref, mix).mean())
    geo = np.mean(others, 0)
    geo = geo - np.logaddexp.reduce(geo, -1, keepdims=True)
    res["geometric mean of seeds 1-4 vs attn_s0"] = float(kl_rows(ref, geo).mean())
    # data cross-entropy of each (next tokens of the windows)
    nxt = np.asarray(W[:, 1:257], np.int64).reshape(-1)
    ce = lambda lp: float(-lp[np.arange(len(nxt)), nxt].mean())
    res["data CE"] = {**{m: ce(lp) for m, lp in lps.items()}, "program": ce(lq)}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "seed_floor.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
