"""Per-model numbers for a ladder model: dev loss, KL to the unigram, bigram and trigram baselines, and the seed
floor (KL to the probability average of the other four seeds of the same rung and arm).

    .venv/bin/python -m analysis.ladder.eval --rung L1 --arm ln --seed 0
    .venv/bin/python -m analysis.ladder.eval --check_l10        # cross-check of this module's KL against the harness

The account scoring that used to live here was removed with decision 23 (git tag pre-proofs).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.ladder.common import REPO, SEEDS, kl_pos, load, log_mean_exp, model_dir, model_logprobs, stat, windows
from src.train import data_char


def floor_logprobs(rung: str, arm: str, seed: int, W: np.ndarray):
    """Probability average of the other four seeds, or None if they are not all trained."""
    others = [s for s in SEEDS if s != seed]
    if not all(os.path.exists(os.path.join(model_dir(rung, arm, s), "final.pt")) for s in others):
        return None
    return log_mean_exp([model_logprobs(load(rung, arm, s), W) for s in others])


def score(rung: str, arm: str, seed: int, split: str = "dev_full") -> dict:
    W = windows(rung, split)
    tgt = W[:, 1:].astype(np.int64)
    lp_m = model_logprobs(load(rung, arm, seed), W)
    tables = data_char.load_baselines()
    r = {"rung": rung, "arm": arm, "seed": seed, "split": split, "n_windows": len(W),
         "ce_model": float(-np.take_along_axis(lp_m, tgt[..., None], -1).mean())}
    for order, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
        lb = data_char.baseline_logprobs(W, order, tables)
        r[f"kl_{name}"] = stat(kl_pos(lp_m, lb))["mean"]
        r[f"ce_{name}"] = float(-np.take_along_axis(lb, tgt[..., None], -1).mean())
    fl = floor_logprobs(rung, arm, seed, W)
    r["kl_floor"] = stat(kl_pos(lp_m, fl))["mean"] if fl is not None else None
    return r


def check_l10() -> dict:
    """KL of program v1.3 against attn_s0 on dev_fast, through this module's KL and the harness's program runner and
    model. Must reproduce the harness's 0.549659 (logs/iterations.csv, the clean-up verification)."""
    from harness.common import ProgramRunner, eval_windows, load_model as h_load, model_logprobs as h_lp

    W = eval_windows("dev_fast")
    lp_m = h_lp(h_load("attn_s0"), W)
    with ProgramRunner(os.path.join(REPO, "program")) as run:
        lp_p = np.stack([run.predict(w[:256]) for w in W])
    k = stat(kl_pos(lp_m, lp_p))["mean"]
    return {"kl_dev_fast": k, "harness": 0.549659, "agrees": abs(k - 0.549659) < 1e-5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung")
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev_full", choices=["dev_full", "dev_fast"])
    ap.add_argument("--check_l10", action="store_true")
    a = ap.parse_args()
    if a.check_l10:
        print(json.dumps(check_l10(), indent=1))
        return
    print(json.dumps(score(a.rung, a.arm, a.seed, a.split), indent=1))


if __name__ == "__main__":
    main()
