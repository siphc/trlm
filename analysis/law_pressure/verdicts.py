"""Study B's pre-registered outcomes (logs/prereg_law_pressure.md, FINAL): per confirmatory seed the manipulation check,
the held-out law (INTACT / PARTLY BROKEN / BROKEN), the cost (LOW / MODERATE / HIGH) and the cell of the outcome table;
the verdict for arm both; the same classification for arms attn and vote (comparisons); the restoring force.

    .venv/bin/python -m analysis.law_pressure.verdicts [--partial]

Inputs: models/law_pressure/<run>/result.json; in-batch correlations (computed here on train_windows(16, 9300), cached in
analysis/out/law_pressure/measures.json); goal 1's law tools on each model and continuation
(analysis/out/look_write/law_pressure_<run>/arms.json, law_pressure_restore_<run>/arms.json). Writes
analysis/out/law_pressure/verdicts.json and prints a summary. Refuses to run before the FINAL commit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from analysis.rules_common import REPO, load_rules

OUT = os.path.join(REPO, "analysis", "out", "law_pressure")
SEEDS = [1, 2, 3, 4]
CELLS = {("INTACT", "LOW"): "GAMED", ("INTACT", "HIGH"): "PAID WITHOUT BREAKING",
         ("BROKEN", "LOW"): "FREE BREAK", ("BROKEN", "HIGH"): "COSTLY BREAK"}


def final_committed():
    log = subprocess.run(["git", "log", "--format=%s", "--", "logs/prereg_law_pressure.md"], cwd=REPO,
                         capture_output=True, text=True).stdout
    return "FINAL pre-registration" in log


def mean_abs_corr(run, root=os.path.join(REPO, "models", "law_pressure")):
    """Mean |in-batch correlation| over heads on train_windows(16, 9300) (cached)."""
    import torch

    from analysis.fitlib import train_windows
    from src.train.law_pressure import corr_on
    from src.train.train import load_model

    p = os.path.join(OUT, "measures.json")
    cache = json.load(open(p)) if os.path.exists(p) else {}
    if run not in cache:
        p1 = torch.as_tensor(np.exp(np.load(os.path.join(REPO, "data", "fixed_votes", "tables.npz"))["follow"]
                                    .astype(np.float64)), dtype=torch.float32)
        rho = corr_on(load_model(os.path.join(root, run, "final.pt")), train_windows(16, 9300), p1)
        cache[run] = {"rho": rho, "mean_abs": float(np.mean(np.abs(rho)))}
        os.makedirs(OUT, exist_ok=True)
        with open(p, "w") as f:
            json.dump(cache, f, indent=1)
    return cache[run]["mean_abs"]


def law(run):
    """Law strength (mean over heads of the pure-pair correlation with own usefulness) and the mean own-usefulness
    output gain, from goal 1's law tools; None if not computed yet."""
    p = os.path.join(REPO, "analysis", "out", "look_write", f"law_pressure_{run.replace('/', '_')}", "arms.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["heads"]
    return {"strength": float(np.mean([r[h]["pure_pair_corr"]["own"] for h in r])),
            "own_gain": float(np.mean([r[h]["arms"]["own"]["gain"] for h in r]))}


def dev_full(run):
    return json.load(open(os.path.join(REPO, "models", "law_pressure", run, "result.json")))["dev_full_loss"]


def classify(new, twin, ce_new, ce_twin, corr_new, corr_twin):
    """One seed: manipulation check, law state, cost, cell (pre-registered thresholds)."""
    m = corr_new <= 0.2 * corr_twin
    if new["strength"] >= 0.25 and new["own_gain"] >= 0.5 * twin["own_gain"]:
        state = "INTACT"
    elif new["strength"] < 0.10 or new["own_gain"] <= 0.25 * twin["own_gain"]:
        state = "BROKEN"
    else:
        state = "PARTLY BROKEN"
    rel = ce_new / ce_twin - 1
    cost = "HIGH" if rel >= 0.02 else "LOW" if rel < 0.01 else "MODERATE"
    cell = "INCONCLUSIVE" if not m else CELLS.get((state, cost), f"{state}, cost {cost}")
    return {"M": m, "corr_ratio": corr_new / corr_twin, "law": new, "twin_law": twin, "state": state,
            "relative_cost": rel, "cost": cost, "cell": cell}


def verdict(cells):
    for c in CELLS.values():
        if sum(x == c for x in cells) >= 3:
            return c
    return "MIXED"


def recovery(treated, cont_treated, cont_twin):
    den = cont_twin - treated
    return (cont_treated - treated) / den if den > 0 else None


def restoring_verdict(rs):
    ok = [r for r in rs if r is not None]
    if sum(r >= 0.5 for r in ok) >= 3:
        return "FAST"
    if sum(r <= 0.1 for r in ok) >= 3:
        return "NONE"
    return "SLOW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if not final_committed():
        raise SystemExit("the FINAL pre-registration of study B is not committed")
    rules = load_rules("law_pressure")
    lam = rules["R1"]["lambda_star"]
    arms = ["both"] + [x for x in ("attn", "vote") if rules.get("R3", {}).get(x, {}).get("admitted")]
    res = {"lambda": lam, "arms": {}}
    for arm in arms:
        per = {}
        for s in SEEDS:
            run, twin = f"{arm}_l{lam:g}_s{s}", f"both_l0_s{s}"
            if not all(os.path.exists(os.path.join(REPO, "models", "law_pressure", r, "result.json")) for r in (run, twin)):
                continue
            ln, lt = law(run), law(twin)
            if ln is None or lt is None:
                continue
            per[s] = classify(ln, lt, dev_full(run), dev_full(twin), mean_abs_corr(run), mean_abs_corr(twin))
            if arm == "both":
                cn, ct = law(f"restore/{run}"), law(f"restore/{twin}")
                if cn and ct:
                    per[s]["restoring"] = {"law_continued": cn, "twin_law_continued": ct,
                                           "recovery": recovery(ln["strength"], cn["strength"], ct["strength"])}
        r = {"per_seed": per}
        if len(per) == 4:
            r["verdict"] = verdict([p["cell"] for p in per.values()])
            if arm == "both" and all("restoring" in p for p in per.values()):
                r["restoring_force"] = restoring_verdict([p["restoring"]["recovery"] for p in per.values()])
        elif not a.partial:
            raise SystemExit(f"arm {arm}: seeds complete {sorted(per)}; use --partial")
        res["arms"][arm] = r
    with open(os.path.join(OUT, "verdicts.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for arm, r in res["arms"].items():
        print(f"arm {arm}: verdict {r.get('verdict', '(partial)')}; restoring force {r.get('restoring_force', '-')}")
        for s, p in r["per_seed"].items():
            rec = p.get("restoring", {}).get("recovery")
            print(f"  seed {s}: M {'pass' if p['M'] else 'fail'} (ratio {p['corr_ratio']:.2f}), law {p['state']} "
                  f"(strength {p['law']['strength']:.3f} vs {p['twin_law']['strength']:.3f}), cost {p['relative_cost']:+.2%} "
                  f"-> {p['cell']}" + ("" if rec is None else f"; recovery {rec:.2f}"))


if __name__ == "__main__":
    main()
