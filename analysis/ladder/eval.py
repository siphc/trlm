"""Score an account against a ladder model: pass conditions P1 and P2 and the reported per-rung numbers
(logs/ladder/design.md, Completeness criteria).

    .venv/bin/python -m analysis.ladder.eval --rung L1 --arm ln --seed 0 --account ladder/L1_ln/account.py
    .venv/bin/python -m analysis.ladder.eval --rung L1 --arm ln --seed 0             # model and baselines only
    .venv/bin/python -m analysis.ladder.eval --check_l10                             # the harness cross-check

P1 (decision 18): KL(model || account) <= max(KL_floor, 0.01) nats per symbol on dev_full, KL_floor = KL(model ||
    probability average of the other four seeds).
Reported: coverage (KL_bigram - KL_account) / (KL_bigram - KL_floor) (the former P2).
Appends a row to logs/ladder/iterations.csv when an account is scored.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os

import numpy as np

from analysis.ladder.common import (REPO, SEEDS, account_logprobs, file_sha, git_commit, kl_pos, load, load_account,
                                    log_mean_exp, model_dir, model_logprobs, stat, windows)
from src.train import data_char

P1_MIN_THRESHOLD = 0.01
CSV = os.path.join(REPO, "logs", "ladder", "iterations.csv")
FIELDS = ["timestamp", "commit", "account", "account_sha", "rung", "arm", "seed", "split", "n_windows", "kl",
          "kl_se", "top1", "ce_model", "ce_account", "kl_bigram", "kl_trigram", "kl_floor", "coverage", "P1_threshold", "P1"]


def floor_logprobs(rung: str, arm: str, seed: int, W: np.ndarray):
    """Probability average of the other four seeds, or None if they are not all trained."""
    others = [s for s in SEEDS if s != seed]
    if not all(os.path.exists(os.path.join(model_dir(rung, arm, s), "final.pt")) for s in others):
        return None
    return log_mean_exp([model_logprobs(load(rung, arm, s), W) for s in others])


def score(rung: str, arm: str, seed: int, account: str | None, split: str = "dev_full") -> dict:
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
    if account:
        lp_a = account_logprobs(load_account(account), W)
        k = stat(kl_pos(lp_m, lp_a))
        r.update({"account": os.path.relpath(account, REPO), "account_sha": file_sha(account),
                  "kl": k["mean"], "kl_se": k["se"], "top1": float((lp_a.argmax(-1) == lp_m.argmax(-1)).mean()),
                  "ce_account": float(-np.take_along_axis(lp_a, tgt[..., None], -1).mean())})
        if r["kl_floor"] is None:
            r["coverage"], r["P1_threshold"], r["P1"] = None, None, "floor not available (other seeds untrained)"
        else:
            r["coverage"] = (r["kl_bigram"] - k["mean"]) / (r["kl_bigram"] - r["kl_floor"])
            r["P1_threshold"] = max(r["kl_floor"], P1_MIN_THRESHOLD)
            r["P1"] = bool(k["mean"] <= r["P1_threshold"])
    return r


def append(r: dict) -> None:
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**r, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit()})


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
    ap.add_argument("--account")
    ap.add_argument("--split", default="dev_full", choices=["dev_full", "dev_fast"])
    ap.add_argument("--check_l10", action="store_true")
    a = ap.parse_args()
    if a.check_l10:
        print(json.dumps(check_l10(), indent=1))
        return
    r = score(a.rung, a.arm, a.seed, a.account, a.split)
    print(json.dumps(r, indent=1))
    if a.account:
        append(r)


if __name__ == "__main__":
    main()
