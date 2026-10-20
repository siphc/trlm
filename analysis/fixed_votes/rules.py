"""Study A decision rules (logs/prereg_fixed_votes.md), values written to analysis/out/fixed_votes/rules.json.

  .venv/bin/python -m analysis.fixed_votes.rules r2 --device mps   # seconds per fixed-vote step (R2; R3 if > 0.25 s)
  .venv/bin/python -m analysis.fixed_votes.rules r1                # peak LR from the three 50M-token runs
"""
from __future__ import annotations

import argparse
import json
import os

from analysis.rules_common import REPO, device_name, save_rules, seconds_per_step

STUDY = "fixed_votes"
LRS = [1e-3, 2e-3, 4e-3]
TABLES = "next,copy,topic,skip2"


def cmd_r2(device):
    fixed = seconds_per_step(device, variant="fixed", fixed_tables=TABLES)
    plain = seconds_per_step(device, variant="attn")
    res = {"device": device_name(device), "seconds_per_step": {"fixed_votes": fixed, "plain_attn": plain},
           "over_0.25s": fixed > 0.25, "epoch_hours": fixed * 32195 / 3600}
    save_rules(STUDY, {f"R2_{device}": res})
    print(json.dumps(res, indent=1))


def cmd_r1():
    res = {}
    for lr in LRS:
        p = os.path.join(REPO, "models", "fixed_votes", "lr", f"base_s0_50M_lr{lr:g}", "result.json")
        res[f"{lr:g}"] = json.load(open(p))["dev_full_loss"]
    best = min(res, key=res.get)
    out = {"dev_full_after_50M": res, "peak_lr": float(best)}
    save_rules(STUDY, {"R1": out})
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=["r1", "r2"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    {"r1": cmd_r1, "r2": lambda: cmd_r2(a.device)}[a.rule]()


if __name__ == "__main__":
    main()
