"""Collect per-model arms.json files, evaluate the pre-registered predictions P1-P4 and the verdict, and write
analysis/look_write/results.json and analysis/look_write/tables.md (generated tables; RESULTS.md quotes them).

Run: .venv/bin/python -m analysis.look_write.summarize
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.look_write import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIRM = ["attn_s1", "attn_s2", "attn_s3", "attn_s4"]
ALL = ["attn_s0"] + CONFIRM


def load(model_id, tag=""):
    p = os.path.join(C.OUT, model_id, f"arms{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def predictions(r):
    """P1-P3 for one head result; P4 is filled in later (needs all heads)."""
    a, pc = r["arms"], r["pure_pair_corr"]
    own, tt = a["own"]["gain"], a["tt"]["gain"]
    others = {k: v["gain"] for k, v in a.items() if k.startswith("other_")}
    p1 = own >= 0.5 * tt
    worst_ctrl = max(list(others.values()) + [a["perm"]["gain"]])
    p2 = worst_ctrl <= 0.25 * own
    oc = {k: v for k, v in pc.items() if k.startswith("other_")}
    p3 = pc["own"] >= 0.25 and all(pc["own"] > v for v in oc.values())
    return {"P1": bool(p1), "P1_ratio_own_over_tt": own / tt if tt != 0 else None,
            "P2": bool(p2), "P2_max_control_over_own": worst_ctrl / own if own != 0 else None,
            "P3": bool(p3), "P3_own_corr": pc["own"], "P3_max_other_corr": max(oc.values())}


def verdict_of(preds):
    n = len(preds)
    n3 = sum(p[0] and p[1] and p[2] for p in preds)
    nf = sum(not p[0] for p in preds)
    v = "SUPPORTED" if n and n3 >= 0.8 * n else ("REFUTED" if n and nf >= 0.5 * n else "MIXED")
    return {"n": n, "n_P1_P2_P3": n3, "n_P1_fail": nf, "verdict": v}


def paired_se(r):
    """Paired SE (over the 32 eval windows) of the own and TT gains and of the P1 margin own - 0.5 TT."""
    a = r["arms"]
    if "kl_win" not in a["base"]:
        return None
    b, o, t = (np.array(a[k]["kl_win"]) for k in ("base", "own", "tt"))
    n = len(b)
    se = lambda x: float(x.std(ddof=1) / np.sqrt(n))  # noqa: E731
    return {"own_gain": float((b - o).mean()), "own_gain_se": se(b - o), "tt_gain": float((b - t).mean()),
            "tt_gain_se": se(b - t), "P1_margin": float(((b - o) - 0.5 * (b - t)).mean()),
            "P1_margin_se": se((b - o) - 0.5 * (b - t))}


def sensitivity(res, local):
    """Exploratory re-readings of the verdict (the pre-registered verdict is the literal one in summary)."""
    out = {}
    conf = [(m, h) for m, h in local if m in CONFIRM]
    P = lambda m, h: res[m]["heads"][str(h)]["predictions"]  # noqa: E731
    lit = [(P(m, h)["P1"], P(m, h)["P2"], P(m, h)["P3"]) for m, h in conf]
    out["literal"] = verdict_of(lit)
    # (a) P1 counted as failed when the exact-TT ceiling itself has a negative gain
    pos = [(P(m, h)["P1"] and res[m]["heads"][str(h)]["arms"]["tt"]["gain"] > 0, P(m, h)["P2"], P(m, h)["P3"])
           for m, h in conf]
    out["P1_fails_if_TT_gain_negative"] = verdict_of(pos)
    # (b) other ceilings (extra_arms.py): TT only at d = 1..7 (round-2 structure), or the best of the three forms
    extra = {m: load(m, "_extra") for m in ALL}
    for name in ("tt_d17", "tt_self", "best_ceiling"):
        rows, detail = [], {}
        for m, h in conf:
            if not extra[m] or (name == "tt_self" and "tt_self" not in extra[m]["heads"][str(h)]):
                continue
            a = res[m]["heads"][str(h)]["arms"]
            e = extra[m]["heads"][str(h)]
            ceil = e[name]["gain"] if name != "best_ceiling" else max([a["tt"]["gain"]] +
                                                                      [v["gain"] for v in e.values()])
            p1 = a["own"]["gain"] >= 0.5 * ceil
            rows.append((p1, P(m, h)["P2"], P(m, h)["P3"]))
            detail[f"{m}/h{h}"] = {"ceiling_gain": ceil, "ratio_own_over_ceiling": a["own"]["gain"] / ceil, "P1": p1}
        if rows:
            out[f"ceiling_{name}"] = {**verdict_of(rows), "heads": detail}
    # (c) LOCAL by the per-query (renormalised) mean distance instead of the pooled one
    alt = [(m, int(h)) for m in CONFIRM if res[m] for h, r in res[m]["heads"].items()
           if r["mean_dist"]["per_query"] <= C.LOCAL_MAX_DIST]
    rows = []
    for m, h in alt:
        p = predictions(res[m]["heads"][str(h)])
        rows.append((p["P1"], p["P2"], p["P3"]))
    out["LOCAL_by_per_query_distance"] = {**verdict_of(rows), "heads": [f"{m}/h{h}" for m, h in alt]}
    # (d) P1-P3 on RECALL heads (no prediction; descriptive)
    rec = {}
    for m in ALL:
        if not res[m]:
            continue
        for h, r in res[m]["heads"].items():
            if r["kind"] == "RECALL":
                rec[f"{m}/h{h}"] = predictions(r)
    out["RECALL_heads_P1_P3_descriptive"] = rec
    return out


def main():
    res = {m: load(m) for m in ALL}
    missing = [m for m, r in res.items() if r is None]
    if missing:
        print("missing:", missing)
    local = [(m, int(h)) for m in ALL if res[m] for h, r in res[m]["heads"].items() if r["kind"] == "LOCAL"]
    for m, h in local:
        res[m]["heads"][str(h)]["predictions"] = predictions(res[m]["heads"][str(h)])
    # P4: own-UOV coefficients vs their median over all LOCAL heads of seeds 0-4
    med = {}
    for c in ("c1", "c2"):
        vals = [res[m]["heads"][str(h)]["arms"]["own"]["coef"][c] for m, h in local]
        med[c] = float(np.median(vals))
    for m, h in local:
        p = res[m]["heads"][str(h)]["predictions"]
        co = res[m]["heads"][str(h)]["arms"]["own"]["coef"]
        for c in ("c1", "c2"):
            ratio = co[c] / med[c]
            p[f"P4_{c}_ratio_to_median"] = ratio
            p[f"P4_{c}"] = bool(1 / 1.5 <= ratio <= 1.5)
        p["P4"] = p["P4_c1"] and p["P4_c2"]
    conf = [(m, h) for m, h in local if m in CONFIRM]
    allp = [res[m]["heads"][str(h)]["predictions"] for m, h in conf]
    n = len(allp)
    n_all3 = sum(p["P1"] and p["P2"] and p["P3"] for p in allp)
    n_p1_fail = sum(not p["P1"] for p in allp)
    if n and n_all3 >= 0.8 * n:
        verdict = "SUPPORTED"
    elif n and n_p1_fail >= 0.5 * n:
        verdict = "REFUTED"
    else:
        verdict = "MIXED"
    summary = {"verdict": verdict, "n_local_heads_confirmatory": n, "n_P1_P2_P3_all_hold": n_all3,
               "n_P1_fail": n_p1_fail, "n_P1": sum(p["P1"] for p in allp), "n_P2": sum(p["P2"] for p in allp),
               "n_P3": sum(p["P3"] for p in allp), "n_P4": sum(p["P4"] for p in allp),
               "P4_medians_local_seeds0_4": med, "local_heads": [f"{m}/h{h}" for m, h in local]}
    summary["sensitivity"] = sensitivity(res, local)
    summary["paired_se"] = {f"{m}/h{h}": paired_se(res[m]["heads"][str(h)]) for m in ALL if res[m]
                            for h in range(C.H) if paired_se(res[m]["heads"][str(h)])}
    explo = {}
    for tag in ("_uovband",):
        explo[tag.strip("_")] = {m: load(m, tag) for m in ALL + ["mlp_s0"] if load(m, tag)}
    mlp = load("mlp_s0")
    out = {"summary": summary, "models": res, "exploratory": {"mlp_s0": mlp, **explo}}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_tables(res, local, summary, mlp, explo)
    print(json.dumps(summary, indent=1))


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def _c(coef, name):
    if name in coef:
        return f"{coef[name]:.3f}"
    return f"cb={coef['cb']:.3f}" if (name == "c1" and "cb" in coef) else "-"


def head_rows(res_m, model):
    rows = []
    for h, r in res_m["heads"].items():
        a = r["arms"]
        others = " / ".join(f"{a[k]['gain']:+.4f}" for k in sorted(a) if k.startswith("other_"))
        rows.append(f"| {model} | {h} | {r['kind']} | {r['mean_dist']['pooled']:.2f} | {a['base']['kl']:.4f} | "
                    f"{a['own']['gain']:+.4f} | {others} | {a['perm']['gain']:+.4f} | {a['tt']['gain']:+.4f} | "
                    f"{_c(a['own']['coef'], 'c1')} | {_c(a['own']['coef'], 'c2')} | {a['tt']['coef']['c']:.3f} |")
    return rows


def corr_rows(res_m, model):
    rows = []
    for h, r in res_m["heads"].items():
        pc = r["pure_pair_corr"]
        others = " / ".join(f"{pc[k]:+.3f}" for k in sorted(pc) if k.startswith("other_"))
        rows.append(f"| {model} | {h} | {r['kind']} | {pc['own']:+.3f} | {others} | {pc['perm']:+.3f} | "
                    f"{pc['tt']:+.3f} |")
    return rows


def write_tables(res, local, summary, mlp, explo):
    L = []
    L.append("## Per head: classification, base KL, output-test gains, fitted coefficients\n")
    L.append("Gains = KL(base) - KL(arm), nats/token, exact-model substitution on train_windows(32, 2025). "
             "`others` lists the other heads' UOV arms in increasing head order (the head itself skipped). "
             "c1, c2 = own-UOV coefficients for d = 1 and d = 2..7; c_TT = coefficient of the exact TT term.\n")
    L.append("| model | head | kind | mean dist | base KL | own | others | perm | TT | c1 | c2 | c_TT |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ALL:
        if res[m]:
            L += head_rows(res[m], m)
    L.append("\n## Pure-pair correlations at d = 1 (double-centred, weighted by N_1, pairs with N_1 >= 20)\n")
    L.append("| model | head | kind | own UOV | other heads' UOV | perm | TT |")
    L.append("|---|---|---|---|---|---|---|")
    for m in ALL:
        if res[m]:
            L += corr_rows(res[m], m)
    L.append("\n## Predictions per LOCAL head\n")
    L.append("| model/head | P1 (own/TT gain) | P2 (max control/own) | P3 (own corr; max other) | "
             "P4 c1 (ratio to median) | P4 c2 (ratio) |")
    L.append("|---|---|---|---|---|---|")
    for m, h in local:
        p = res[m]["heads"][str(h)]["predictions"]
        tag = "" if m != "attn_s0" else " (cross-check)"
        L.append(f"| {m}/h{h}{tag} | {'yes' if p['P1'] else 'NO'} ({fmt(p['P1_ratio_own_over_tt'], 2)}) | "
                 f"{'yes' if p['P2'] else 'NO'} ({fmt(p['P2_max_control_over_own'], 2)}) | "
                 f"{'yes' if p['P3'] else 'NO'} ({p['P3_own_corr']:.3f}; {p['P3_max_other_corr']:.3f}) | "
                 f"{'yes' if p['P4_c1'] else 'NO'} ({p['P4_c1_ratio_to_median']:.2f}) | "
                 f"{'yes' if p['P4_c2'] else 'NO'} ({p['P4_c2_ratio_to_median']:.2f}) |")
    s = summary
    L.append(f"\nMedians over LOCAL heads of seeds 0-4: c1 = {s['P4_medians_local_seeds0_4']['c1']:.3f}, "
             f"c2 = {s['P4_medians_local_seeds0_4']['c2']:.3f}.\n")
    L.append(f"Confirmatory LOCAL heads (seeds 1-4): {s['n_local_heads_confirmatory']}; P1 holds {s['n_P1']}, "
             f"P2 {s['n_P2']}, P3 {s['n_P3']}, all of P1-P3 {s['n_P1_P2_P3_all_hold']}, P4 {s['n_P4']}. "
             f"**Verdict: {s['verdict']}.**\n")
    if mlp:
        L.append("\n## Exploratory: mlp_s0 (OV through the direct path only)\n")
        L.append("| model | head | kind | mean dist | base KL | own | others | perm | TT | c1 | c2 | c_TT |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        L += head_rows(mlp, "mlp_s0")
        L.append("\n| model | head | kind | own UOV | other heads' UOV | perm | TT |")
        L.append("|---|---|---|---|---|---|---|")
        L += corr_rows(mlp, "mlp_s0")
    for name, d in explo.items():
        if d:
            L.append(f"\n## Exploratory: {name} (UOV aggregated over the distance band, see RESULTS.md)\n")
            L.append("| model | head | kind | mean dist | base KL | own | others | perm | TT | c1 | c2 | c_TT |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for m, r in d.items():
                L += head_rows(r, m)
    with open(os.path.join(HERE, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
