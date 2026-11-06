"""The coverage measure (logs/decisions.md item 14; logs/coverage.md): how much of the way from a simple baseline to
the seed floor the program gets, per seed. Analysis only; TRAIN windows (analysis/mech_eval.EVAL_SEED, as
ensemble_target.py and seed_floor.py).

For seed k = 0..4 (attn_s{k}), with KL(seed k || x) per token:
    floor      x = the probability average of the other four seeds (what any description that does not know seed k's
                 own training run could at best be expected to reach; a single sibling seed is reported too)
    baselines  the harness's unigram, bigram and trigram baselines (baselines/)
    program    the current program/ (v1.3 at the time of writing)
    coverage   (KL_bigram - KL_program) / (KL_bigram - KL_floor); also against the trigram

Also KL(ENS5 || program) and the seeds' leave-one-out KL(mean of the other four || seed k) (what the five seeds
share, from the consensus side), with standard errors over windows.

    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python -m analysis.program_gap.coverage [--n 64]
Writes analysis/out/program_gap/coverage.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, program
from analysis.program_gap.bottleneck import OUT
from harness.common import load_model, model_logprobs

SEEDS = [f"attn_s{k}" for k in range(5)]
BASELINES = ("unigram", "bigram", "trigram")


def kl_win(lp, lq):
    """Per-window mean KL(p || q), (n,)."""
    p = np.exp(lp.astype(np.float64))
    return (p * (lp.astype(np.float64) - lq.astype(np.float64))).sum(-1).mean(-1)


def stat(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}


def log_mean_exp(lps):
    return np.log(np.mean([np.exp(lp.astype(np.float64)) for lp in lps], 0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    W = train_windows(a.n, seed=EVAL_SEED)
    X = np.asarray(W[:, :256], np.int64)
    lps = {m: model_logprobs(load_model(m), W).astype(np.float32) for m in SEEDS}
    P = program()
    prog = np.stack([P.predict(x) for x in X]).astype(np.float32)
    base = {b: np.stack([importlib.import_module(f"baselines.{b}").predict(x) for x in X]).astype(np.float32)
            for b in BASELINES}
    res = {"n_windows": a.n, "windows": f"fitlib.train_windows({a.n}, seed={EVAL_SEED})", "per_seed": {}}
    for m in SEEDS:
        others = [lps[o] for o in SEEDS if o != m]
        floor = log_mean_exp(others)
        r = {"floor": stat(kl_win(lps[m], floor)), "program": stat(kl_win(lps[m], prog)),
             "siblings": {o: float(kl_win(lps[m], lps[o]).mean()) for o in SEEDS if o != m}}
        for b in BASELINES:
            r[b] = stat(kl_win(lps[m], base[b]))
        for b in ("bigram", "trigram"):
            kb, kp, kf = r[b]["mean"], r["program"]["mean"], r["floor"]["mean"]
            r[f"coverage_vs_{b}"] = (kb - kp) / (kb - kf)
            r[f"coverage_vs_{b}_single_sibling"] = (kb - kp) / (kb - float(np.mean(list(r["siblings"].values()))))
        r["consensus_loo"] = stat(kl_win(floor, lps[m]))           # KL(mean of the other four || seed k)
        res["per_seed"][m] = r
        print(f"{m}: bigram {r['bigram']['mean']:.3f} trigram {r['trigram']['mean']:.3f} program "
              f"{r['program']['mean']:.3f} floor {r['floor']['mean']:.3f} (sibling mean "
              f"{np.mean(list(r['siblings'].values())):.3f}); coverage vs bigram {r['coverage_vs_bigram']:.3f}, "
              f"vs trigram {r['coverage_vs_trigram']:.3f}", flush=True)
    ens = log_mean_exp([lps[m] for m in SEEDS])
    res["ens5_to_program"] = stat(kl_win(ens, prog))
    res["ens5_to_bigram"] = stat(kl_win(ens, base["bigram"]))
    res["mean"] = {k: float(np.mean([res["per_seed"][m][k] for m in SEEDS]))
                   for k in ("coverage_vs_bigram", "coverage_vs_trigram", "coverage_vs_bigram_single_sibling")}
    for k in ("program", "floor", "bigram", "trigram", "unigram", "consensus_loo"):
        res["mean"][k] = float(np.mean([res["per_seed"][m][k]["mean"] for m in SEEDS]))
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=1))
    with open(os.path.join(OUT, "coverage.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
