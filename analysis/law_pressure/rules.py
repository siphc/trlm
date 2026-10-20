"""Study B decision rules (logs/prereg_law_pressure.md), values written to analysis/out/law_pressure/rules.json.

  .venv/bin/python -m analysis.law_pressure.rules r2 --device mps   # penalised vs plain step time -> law_windows
  .venv/bin/python -m analysis.law_pressure.rules r1                # lambda* from the development grid
  .venv/bin/python -m analysis.law_pressure.rules r3                # arms attn and vote admitted?

R1 and R3 measure the in-batch correlation (src/train/law_pressure.corr_on) on train_windows(16, 9300) for each
development model (models/law_pressure/dev/<arm>_l<lambda>_s0_100M; lambda 0 is the control).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.rules_common import REPO, device_name, load_rules, save_rules, seconds_per_step

STUDY = "law_pressure"
GRID = [1, 3, 10, 30, 100]
DEV = os.path.join(REPO, "models", "law_pressure", "dev")


def cmd_r2(device):
    """R2: the penalty uses the first 16 windows of each batch if a penalised step takes > 1.6 x a plain step."""
    plain = seconds_per_step(device, variant="attn")
    pen = seconds_per_step(device, variant="attn", law_arm="both", lambda_law=1.0)
    res = {"device": device_name(device), "seconds_per_step": {"plain": plain, "penalised": pen},
           "ratio": pen / plain, "law_windows": 16 if pen / plain > 1.6 else 0}
    save_rules(STUDY, {"R2": res})
    print(json.dumps(res, indent=1))


def correlations(name):
    """Mean |in-batch correlation| over heads (and per head) for one development model; cached in rules.json."""
    import torch

    from analysis.fitlib import train_windows
    from src.train.law_pressure import corr_on
    from src.train.train import load_model

    cache = load_rules(STUDY).get("corr", {})
    if name in cache:
        return cache[name]
    p1 = torch.as_tensor(np.exp(np.load(os.path.join(REPO, "data", "fixed_votes", "tables.npz"))["follow"]
                                .astype(np.float64)), dtype=torch.float32)
    rho = corr_on(load_model(os.path.join(DEV, name, "final.pt")), train_windows(16, 9300), p1)
    ce = json.load(open(os.path.join(DEV, name, "result.json")))["dev_full_loss"]
    out = {"rho": rho, "mean_abs": float(np.mean(np.abs(rho))), "dev_full": ce}
    cache[name] = out
    save_rules(STUDY, {"corr": cache})
    return out


def cmd_r1():
    """R1: lambda* = the smallest grid lambda whose mean |correlation| is at most 20% of the lambda = 0 control's."""
    ctrl = correlations("both_l0_s0_100M")
    rows = {lam: correlations(f"both_l{lam:g}_s0_100M") for lam in GRID}
    ok = [lam for lam in GRID if rows[lam]["mean_abs"] <= 0.2 * ctrl["mean_abs"]]
    res = {"control": ctrl, "grid": {f"{lam:g}": {**r, "relative_to_control": r["mean_abs"] / ctrl["mean_abs"],
                                                  "cost": r["dev_full"] / ctrl["dev_full"] - 1}
                                     for lam, r in rows.items()},
           "lambda_star": min(ok) if ok else max(GRID), "met": bool(ok)}
    save_rules(STUDY, {"R1": res})
    print(json.dumps({k: v for k, v in res.items() if k != "control"}, indent=1))


def cmd_r3():
    """R3: arms attn and vote enter the confirmatory runs only if their run at lambda* meets R1's 20% criterion."""
    rules = load_rules(STUDY)
    lam, ctrl = rules["R1"]["lambda_star"], rules["R1"]["control"]["mean_abs"]
    res = {}
    for arm in ("attn", "vote"):
        r = correlations(f"{arm}_l{lam:g}_s0_100M")
        res[arm] = {**r, "relative_to_control": r["mean_abs"] / ctrl, "admitted": r["mean_abs"] <= 0.2 * ctrl}
    save_rules(STUDY, {"R3": res})
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=["r1", "r2", "r3"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    {"r1": cmd_r1, "r2": lambda: cmd_r2(a.device), "r3": cmd_r3}[a.rule]()


if __name__ == "__main__":
    main()
