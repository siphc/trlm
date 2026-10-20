"""Study A's pre-registered tests (logs/prereg_fixed_votes.md, FINAL) from the law.json of each confirmatory model
(analysis/fixed_votes/law.py): H1 (the law describes learned attention), H2 (stronger than in normal models), H3
(changing head 3's table moves where it looks), plus the separately reported quantities.

    .venv/bin/python -m analysis.fixed_votes.verdicts [--partial]

Writes analysis/out/fixed_votes/verdicts.json. Refuses to run before the FINAL commit. Conventions (FINAL):
- heads are named by their table (base: next, copy, topic, skip2; shift: next, copy, topic, skip3);
- "another table" in P2' and P3' = the tables of the model's other heads (the alt_ arm is used only in H3b);
- a head type is active in a seed if removing its vote raises dev_fast CE by >= 0.01 (R4); a head type inactive in 2
  or more seeds is excluded from H1/H2 (reported); in the other seeds an inactive head counts as not meeting the tests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from analysis.rules_common import REPO

OUT = os.path.join(REPO, "analysis", "out", "fixed_votes")
SEEDS = [1, 2, 3, 4]
TT_MIN = 0.01


def final_committed():
    log = subprocess.run(["git", "log", "--format=%s", "--", "logs/prereg_fixed_votes.md"], cwd=REPO,
                         capture_output=True, text=True).stdout
    return "FINAL pre-registration" in log


def head_tests(law, h):
    """P1', P2', P3' and the H2 ratio for head h of one model's law.json."""
    hr = law["heads"][str(h)]
    arms, own = hr["arms"], hr["arms"]["own"]["gain"]
    tt, tts = arms["tt"]["gain"], arms["tt_self"]["gain"]
    others = [k for k in arms if k.startswith("other_")]
    p1 = tt > TT_MIN and own >= 0.5 * tt
    p2 = own > 0 and all(arms[k]["gain"] <= 0.25 * own for k in others + ["perm"])
    table = hr["table"]
    if table in ("next", "skip2"):
        pc = hr["pure_pair_corr"]
        p3 = pc[table] >= 0.25 and all(pc[table] > pc[k[len("other_"):]] for k in others)
    else:
        p3 = True                                              # P3' applies to the locally looking heads only
    return {"table": table, "active": law["active"][str(h)], "own_gain": own, "tt_gain": tt, "tt_self_gain": tts,
            "P1": p1, "P2": p2, "P3": p3, "follows": bool(law["active"][str(h)] and p1 and p2 and p3),
            "ratio_tt_self": own / tts if tts > TT_MIN else None}


def h1_h2(laws):
    """laws: {seed: law.json of the base model}. Per head type: follows the law (H1) and stronger (H2)."""
    types = {}
    for s, law in laws.items():
        for h in range(len(law["tables"])):
            t = head_tests(law, h)
            types.setdefault(t["table"], {})[s] = t
    out = {}
    for table, per in types.items():
        inactive = sum(not t["active"] for t in per.values())
        out[table] = {"per_seed": per, "inactive_seeds": inactive, "excluded": inactive >= 2,
                      "follows_law": sum(t["follows"] for t in per.values()) >= 3,
                      "stronger": sum(t["active"] and t["ratio_tt_self"] is not None and t["ratio_tt_self"] > 0.46
                                      for t in per.values()) >= 3}
    active = [t for t, r in out.items() if not r["excluded"]]
    n = sum(out[t]["follows_law"] for t in active)
    h1 = "SUPPORTED" if active and n == len(active) else "PARTIAL" if n >= 1 else "NOT SUPPORTED"
    return h1, out


def predicted_peak(table, beta):
    prof = json.load(open(os.path.join(OUT, "usefulness", "profile.json")))["mean_usefulness"][table]
    vals = {d: prof[str(d)] for d in range(1, 8)}
    return (max if beta >= 0 else min)(vals, key=vals.get)


def h3(base, shift):
    """base, shift: {seed: law.json}. (a) head 3's attention peak at the predicted distance; (b) crossover."""
    res = {}
    for name, laws, own, alt in (("base", base, "skip2", "skip3"), ("shift", shift, "skip3", "skip2")):
        per = {}
        for s, law in laws.items():
            beta = law["beta"][3]
            pred = predicted_peak(own, beta)
            arms = law["heads"]["3"]["arms"]
            per[s] = {"peak": law["peak_distance"]["3"], "predicted": pred, "a": law["peak_distance"]["3"] == pred,
                      "own_gain": arms["own"]["gain"], "alt_gain": arms[f"alt_{alt}"]["gain"],
                      "b": arms["own"]["gain"] > arms[f"alt_{alt}"]["gain"], "beta3": beta}
        res[name] = {"per_seed": per, "a": sum(p["a"] for p in per.values()) >= 3,
                     "b": sum(p["b"] for p in per.values()) >= 3}
    a = res["base"]["a"] and res["shift"]["a"]
    b = res["base"]["b"] and res["shift"]["b"]
    return ("SUPPORTED" if a and b else "PARTIAL" if a or b else "NOT SUPPORTED"), {"a": a, "b": b, **res}


def load_laws(variant):
    out = {}
    for s in SEEDS:
        p = os.path.join(OUT, f"{variant}_s{s}", "law.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def cost(variant):
    out = {}
    for s in SEEDS:
        p = os.path.join(REPO, "models", "fixed_votes", f"{variant}_s{s}", "result.json")
        if not os.path.exists(p):
            continue
        ce = json.load(open(p))["dev_full_loss"]
        row = {"dev_full": ce,
               "vs_attn": ce - json.load(open(os.path.join(REPO, "models", "final", f"attn_s{s}", "result.json")))["dev_full_loss"]}
        tw = os.path.join(REPO, "models", "law_pressure", f"both_l0_s{s}", "result.json")
        if os.path.exists(tw):
            row["vs_same_hardware_lambda0"] = ce - json.load(open(tw))["dev_full_loss"]
        out[s] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if not final_committed():
        raise SystemExit("the FINAL pre-registration of study A is not committed")
    base, shift = load_laws("base"), load_laws("shift")
    if (len(base) < 4 or len(shift) < 4) and not a.partial:
        raise SystemExit(f"seeds with law.json: base {sorted(base)}, shift {sorted(shift)}; use --partial")
    res = {"cost": {"base": cost("base"), "shift": cost("shift")}}
    if base:
        res["H1"], res["per_head_type"] = h1_h2(base)
        res["H2"] = {t: ("STRONGER" if r["stronger"] else "NOT STRONGER") for t, r in res["per_head_type"].items()
                     if not r["excluded"]}
        res["coupling"] = {t: {s: base[s]["heads"][str(h)]["arms"]["own"]["coef"] for s in base}
                           for h, t in enumerate(next(iter(base.values()))["tables"])}
        res["beta"] = {s: base[s]["beta"] for s in base}
    if base and shift:
        res["H3"], res["H3_detail"] = h3(base, shift)
    with open(os.path.join(OUT, "verdicts.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("H1", "H2", "H3") if k in res}, indent=1))


if __name__ == "__main__":
    main()
