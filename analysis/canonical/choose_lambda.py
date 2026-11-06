"""Rule B1 of logs/study_C_readable/prereg_canonical_addendum.md: each control's lambda, chosen on development seed 5 (one epoch) so
that its relative cross-entropy cost matches its arm's confirmatory cost (arm O or I at lambda 0.3, mean over seeds
1-4 against the twins attn_s1..s4).

  - Costs: dev_full CE of models/readable/dev/<ctl>_l<lam>_s5 relative to models/readable/dev/none_s5.
  - Grid: lambda in {0.1, 0.3, 1, 3}; the development queue runs 0.3 and 1 first.
  - If the target cost lies between two run grid values with increasing cost: interpolate log(lambda) linearly in
    cost, round to two significant digits.
  - If it lies below the cost at the smallest run value (> 0.1): ask for a run at the next smaller grid value; above
    the largest (< 3): the next larger. At the end of the grid: take the end value.
  - If the costs are not increasing in lambda: take the run value whose cost is closest to the target.

    .venv/bin/python -m analysis.canonical.choose_lambda      # prints NEED <ctl> <lam> or the chosen values
Writes analysis/out/canonical/rules.json ("B1").
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from analysis.canonical.convergence import OUT
from analysis.readable_training.common import REPO

GRID = (0.1, 0.3, 1.0, 3.0)
DEV_SEED = 5


def ce(path):
    p = os.path.join(path, "result.json")
    return json.load(open(p))["dev_full_loss"] if os.path.exists(p) else None


def target_cost(arm):
    c = [ce(os.path.join(REPO, "models", "readable", f"{arm}_l0.3_s{k}"))
         / ce(os.path.join(REPO, "models", "final", f"attn_s{k}")) - 1 for k in (1, 2, 3, 4)]
    return float(np.mean(c))


def choose(costs, target):
    """costs: {lambda: relative cost} of the runs done. Returns ("need", lam) or ("chosen", lam, how)."""
    lams = sorted(costs)
    cs = [costs[l] for l in lams]
    if any(b <= a for a, b in zip(cs, cs[1:])):
        best = min(lams, key=lambda l: abs(costs[l] - target))
        return ("chosen", best, "costs not increasing: closest run")
    if target < cs[0]:
        smaller = [g for g in GRID if g < lams[0]]
        return ("need", smaller[-1]) if smaller else ("chosen", lams[0], "below the grid: smallest value")
    if target > cs[-1]:
        larger = [g for g in GRID if g > lams[-1]]
        return ("need", larger[0]) if larger else ("chosen", lams[-1], "above the grid: largest value")
    for (l0, c0), (l1, c1) in zip(zip(lams, cs), zip(lams[1:], cs[1:])):
        if c0 <= target <= c1:
            f = (target - c0) / (c1 - c0)
            lam = math.exp(math.log(l0) + f * (math.log(l1) - math.log(l0)))
            return ("chosen", float(f"{lam:.2g}"), f"interpolated between {l0:g} and {l1:g}")
    raise AssertionError("unreachable")


def main():
    base = ce(os.path.join(REPO, "models", "readable", "dev", f"none_s{DEV_SEED}"))
    assert base is not None, "the lambda = 0 development run (none_s5) is not finished"
    res, need = {"dev_seed": DEV_SEED, "none_dev_full": base}, []
    for ctl, arm in (("Octl", "O"), ("Ictl", "I")):
        costs = {}
        for lam in GRID:
            v = ce(os.path.join(REPO, "models", "readable", "dev", f"{ctl}_l{lam:g}_s{DEV_SEED}"))
            if v is not None:
                costs[lam] = v / base - 1
        t = target_cost(arm)
        r = {"target_cost": t, "costs": {f"{k:g}": v for k, v in costs.items()}}
        out = choose(costs, t) if costs else ("need", 0.3)
        if out[0] == "need":
            need.append(f"{ctl} {out[1]:g}")
            r["need"] = out[1]
        else:
            r["lambda"], r["how"] = out[1], out[2]
        res[ctl] = r
    if not need:
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, "rules.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old["B1"] = res
        with open(p, "w") as f:
            json.dump(old, f, indent=1)
    print(json.dumps(res, indent=1))
    for n in need:
        print("NEED", n)


if __name__ == "__main__":
    main()
