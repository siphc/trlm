"""Predictions D1-D5 and the per-family verdicts of logs/prereg_deep_look_write.md, from the arms files.

    .venv/bin/python -m analysis.look_write_deep.summarize_deep --theta 0.2 --c1 0.2426 --c2 0.2710 [--primary ltotal]
        [--models attn2L_s1 attn2L_s2 ... attn4L_s4]

Families: F1 = layer 0 of every multi-block model; F2 = the last layer; F3 = middle layers of 4-block models.
Only confirmatory heads (classes.json) count. Rules:
  D1  ratio-evaluable (oracle gain >= max(0.01, 3 paired SE)): own gain >= theta * oracle gain;
      otherwise (P1-alt) own gain >= 0.15 * base KL
  D2  max(gains of other heads that are not near-duplicates, shuffled gain) <= 0.25 * own gain
  D3  own pure-pair correlation >= 0.25 and above every non-duplicate other head's
  D4  (F1, F3) primary gain >= DV gain - 2 SE for every head; and primary >= DV + 2 SE for >= 75% of the heads whose
      primary and DV tables (d = 1) correlate below 0.5
  D5  (separate) own coefficients c1, c2 within a factor 1.5 of the one-layer medians
Verdict per family: SUPPORTED if D1-D3 hold for >= 80% of its heads; REFUTED if D1 fails for >= 50%; otherwise MIXED;
INCONCLUSIVE with fewer than 4 heads. Writes analysis/out/look_write_deep/summary_<primary>.json
(the committed run used the rules.json values shown above).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.look_write_deep import common_deep as C


def paired(base_win, arm_win):
    d = np.asarray(base_win) - np.asarray(arm_win)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def family(layer, n_blocks):
    if layer == 0:
        return "F1"
    if layer == n_blocks - 1:
        return "F2"
    return "F3"


def head_verdict(hd, primary, theta, c1, c2):
    arms = hd["arms"]
    base = arms["base"]
    own = arms[f"own_{primary}"]
    own_gain, own_se = paired(base["kl_win"], own["kl_win"])
    oqc_gain, oqc_se = paired(base["kl_win"], arms["oqc"]["kl_win"])
    evaluable = oqc_gain >= max(0.01, 3 * oqc_se)
    d1 = own_gain >= theta * oqc_gain if evaluable else own_gain >= 0.15 * base["kl"]
    dups = set(hd["near_duplicates"])
    others = {k: v["gain"] for k, v in arms.items() if k.startswith("other_") and k not in dups}
    controls = list(others.values()) + [arms["shuffled"]["gain"]]
    d2 = own_gain > 0 and max(controls) <= 0.25 * own_gain
    corr = hd["pure_pair_corr"]
    other_corr = [v for k, v in corr.items() if k != "own" and k not in dups]
    d3 = corr["own"] >= 0.25 and all(corr["own"] > v for v in other_corr)
    dv_gain, _ = paired(base["kl_win"], arms["own_dv"]["kl_win"])
    diff_mean, diff_se = paired(arms["own_dv"]["kl_win"], own["kl_win"])     # primary - dv gain, paired
    co = own["coef"]
    d5 = all(np.isfinite(x) and x > 0 and 1 / 1.5 <= x / ref <= 1.5 for x, ref in ((co["c1"], c1), (co["c2"], c2)))
    return {"own_gain": own_gain, "own_se": own_se, "oqc_gain": oqc_gain, "oqc_se": oqc_se, "ratio_evaluable": evaluable,
            "D1": bool(d1), "max_control": max(controls), "D2": bool(d2), "corr_own": corr["own"],
            "max_other_corr": max(other_corr) if other_corr else None, "D3": bool(d3), "dv_gain": dv_gain,
            "primary_minus_dv": diff_mean, "primary_minus_dv_se": diff_se,
            "readout_corr_d1": hd["readout_table_corr_d1"]["dv"], "coef": co, "D5": bool(d5),
            "near_duplicates": sorted(dups)}


def run(models, primary, theta, c1, c2):
    fam = {"F1": [], "F2": [], "F3": []}
    rows = {}
    for mid in models:
        a = json.load(open(os.path.join(C.model_dir(mid), f"arms_{primary}.json")))
        for key, hd in a["heads"].items():
            if not hd["class"]["confirmatory"]:
                continue
            v = head_verdict(hd, primary, theta, c1, c2)
            f = family(hd["class"]["layer"], a["n_blocks"])
            v["family"] = f
            rows[f"{mid}/{key}"] = v
            fam[f].append(f"{mid}/{key}")
    verdicts = {}
    for f, keys in fam.items():
        n = len(keys)
        if n == 0:
            verdicts[f] = {"n": 0, "verdict": "INCONCLUSIVE"}
            continue
        all3 = sum(rows[k]["D1"] and rows[k]["D2"] and rows[k]["D3"] for k in keys)
        d1_fail = sum(not rows[k]["D1"] for k in keys)
        v = ("INCONCLUSIVE" if n < 4 else "SUPPORTED" if all3 >= 0.8 * n else "REFUTED" if d1_fail >= 0.5 * n
             else "MIXED")
        verdict = {"n": n, "D1-D3": all3, "D1": n - d1_fail, "D2": sum(rows[k]["D2"] for k in keys),
                   "D3": sum(rows[k]["D3"] for k in keys), "D5": sum(rows[k]["D5"] for k in keys), "verdict": v}
        if f in ("F1", "F3"):
            not_worse = sum(rows[k]["primary_minus_dv"] >= -2 * rows[k]["primary_minus_dv_se"] for k in keys)
            differ = [k for k in keys if rows[k]["readout_corr_d1"] < 0.5]
            better = sum(rows[k]["primary_minus_dv"] >= 2 * rows[k]["primary_minus_dv_se"] for k in differ)
            verdict["D4"] = {"not_worse_than_dv": not_worse, "n": n, "tables_differ": len(differ),
                             "better_where_differ": better,
                             "holds": not_worse == n and (not differ or better >= 0.75 * len(differ))}
        verdicts[f] = verdict
    headline = verdicts["F1"]["verdict"] == "SUPPORTED" and verdicts["F2"]["verdict"] == "SUPPORTED"
    out = {"primary": primary, "theta": theta, "c1_ref": c1, "c2_ref": c2, "families": verdicts,
           "headline_generalises": headline, "heads": rows}
    with open(os.path.join(C.OUT, f"summary_{primary}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"families": verdicts, "headline_generalises": headline}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[f"attn{L}L_s{s}" for L in (2, 4) for s in (1, 2, 3, 4)])
    ap.add_argument("--primary", default="ltotal")
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--c1", type=float, required=True)
    ap.add_argument("--c2", type=float, required=True)
    a = ap.parse_args()
    run(a.models, a.primary, a.theta, a.c1, a.c2)
