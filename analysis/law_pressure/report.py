"""Study B, reported (not tested) per-head numbers for logs/law_pressure_results.md: "where the work goes" (prereg,
"Deviations and clarifications"). Reads goal 1's law tools output (analysis/out/look_write/law_pressure_<run>/arms.json)
for the twins, the penalised models and both continuations, seeds 1-4.

Per model and head: head kind; pure-pair correlation of attention with own usefulness, with the other heads'
usefulness (largest absolute value: head-specificity) and with the shuffled control; own-usefulness output gain; the
token-token ceiling gain (what exact pair attention adds over the base rule); the base rule's pattern cross-entropy
(eval windows), i.e. how much of attention position and spelling explain.

    .venv/bin/python -m analysis.law_pressure.report        # writes analysis/out/law_pressure/report.json, prints tables
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.rules_common import REPO

OUT = os.path.join(REPO, "analysis", "out", "law_pressure")
SEEDS = (1, 2, 3, 4)
RUNS = {"twin": "both_l0_s{}", "penalised": "both_l1_s{}", "twin, continued": "restore_both_l0_s{}",
        "penalised, continued": "restore_both_l1_s{}"}


def heads(run):
    p = os.path.join(REPO, "analysis", "out", "look_write", f"law_pressure_{run}", "arms.json")
    r = json.load(open(p))["heads"]
    out = {}
    for h, x in r.items():
        pp = x["pure_pair_corr"]
        others = [v for k, v in pp.items() if k.startswith("other_")]
        out[h] = {"kind": x["kind"], "own": pp["own"], "perm": pp["perm"],
                  "max_abs_other": float(max(abs(v) for v in others)),
                  "own_gain": x["arms"]["own"]["gain"], "tt_gain": x["arms"]["tt"]["gain"],
                  "base_pattern_ce": x["arms"]["base"]["pattern_ce_eval"],
                  "tt_pattern_ce": x["arms"]["tt"]["pattern_ce_eval"]}
    return out


def main():
    res = {}
    for label, pat in RUNS.items():
        res[label] = {s: heads(pat.format(s)) for s in SEEDS}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(res, f, indent=1)
    keys = ("own", "own_gain", "tt_gain", "max_abs_other", "perm", "base_pattern_ce")
    print("mean over seeds 1-4, per head (kinds of the twin):")
    print("model".ljust(22), "head kind".ljust(12), "  ".join(k.rjust(15) for k in keys))
    for label in RUNS:
        for h in ("0", "1", "2", "3"):
            vals = [np.mean([res[label][s][h][k] for s in SEEDS]) for k in keys]
            kinds = sorted({res[label][s][h]["kind"] for s in SEEDS})
            print(label.ljust(22), ("/".join(kinds)).ljust(12), "  ".join(f"{v:15.3f}" for v in vals))
    print("\nper seed, mean over heads: own pure-pair corr | own gain | tt gain | max |other|")
    for label in RUNS:
        for s in SEEDS:
            r = res[label][s]
            m = [np.mean([r[h][k] for h in r]) for k in ("own", "own_gain", "tt_gain", "max_abs_other")]
            print(label.ljust(22), s, "  ".join(f"{v:.3f}" for v in m))


if __name__ == "__main__":
    main()
