"""Pre-registered tests and verdicts of the confirmatory runs (logs/prereg_readable_training.md, FINAL; clarifications in
logs/readable_training.md, phase 4, written before any confirmatory output).

    .venv/bin/python -m analysis.readable_training.verdicts [--partial]

Reads the suite.json of each new model (readable_<arm>_l<lambda>_s<seed>) and of its twin attn_s<seed>, rules.json (R3full
lambdas, R4 sigmas), result.json and log.jsonl of the runs, and seed_distances.json (tool e). Writes
analysis/out/readable_training/verdicts.json and prints a markdown summary. Refuses to run before the FINAL commit;
--partial scores the seeds that are complete and gives no verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from analysis.readable_training.apply_rules import tool_values
from analysis.readable_training.common import OUT, REPO, load_rules

SEEDS = [1, 2, 3, 4]
OOD = ["probe_twin", "program_samples", "sent_shuffle", "word_shuffle"]
RHO = {"d4_vote_rho_uniform", "d4_vote_rho_weighted"}                    # higher is better; every other tool is a KL


def final_committed():
    log = subprocess.run(["git", "log", "--format=%s", "--", "logs/prereg_readable_training.md"], cwd=REPO,
                         capture_output=True, text=True).stdout
    return "FINAL pre-registration" in log


def closure(tool, new, base, sigma, literal=False):
    """(base - new) / base for KLs, (new - base) / (1 - base) for correlations; a change smaller than 2 sigma counts as
    0 (clarification 1; literal=True also sets every worsening to 0)."""
    delta = new - base if tool in RHO else base - new                   # positive = toward the program
    if (delta < 2 * sigma) if literal else (abs(delta) < 2 * sigma):
        return 0.0
    return delta / (1 - base) if tool in RHO else delta / base


def ratio(num, den):
    return num / den if den > 0 else None                               # clarification 2


def suite(name):
    p = os.path.join(OUT, name, "suite.json")
    return json.load(open(p)) if os.path.exists(p) else None


def run_dir(arm, lam, seed):
    return os.path.join(REPO, "models", "readable", f"{arm}_l{lam:g}_s{seed}")


def last_eval(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if "m_heldout_out_coarse" in l]
    return rows[-1]


def score_seed(arm, new, twin, sigma, gap, literal=False):
    """Closures, tau, tau_OOD, test M and the three Goodhart flags for one new model against its twin."""
    c = lambda t: closure(t, new[t], twin[t], sigma[t], literal)        # noqa: E731
    cl = {t: c(t) for t in ["b1_harness_kl", "d3_pattern_kl", "d3_pattern_kl_ood_mean", "d4_vote_rho_uniform",
                            "d4_vote_rho_weighted"] + [f"c_kl_{k}" for k in OOD]}
    if arm == "O":
        trained = cl["b1_harness_kl"]
        untrained = (cl["d3_pattern_kl"] + cl["d4_vote_rho_uniform"]) / 2
        tau_ood = ratio(np.mean([cl[f"c_kl_{k}"] for k in OOD]), cl["b1_harness_kl"])
    else:
        trained = (cl["d3_pattern_kl"] + cl["d4_vote_rho_weighted"]) / 2
        untrained = cl["b1_harness_kl"]
        tau_ood = ratio(cl["d3_pattern_kl_ood_mean"], cl["d3_pattern_kl"])
    abl = {h: new[f"d5_ablation_h{h}"] / twin[f"d5_ablation_h{h}"] for h in range(4)}
    flags = {"i_memorisation": gap > 0.02,
             "ii_decorative": any(abl[h] < 0.5 for h in range(3)) or abl[3] > 1.5,
             "iii_refit": new["b2_refit_kl"] - new["b1_harness_kl"] > 0.05}
    return {"closures": cl, "trained_closure": trained, "untrained_closure": untrained,
            "tau": ratio(untrained, trained), "tau_ood": tau_ood, "M": trained >= 0.3,
            "ablation_ratio_to_twin": abl, "memorisation_gap": gap, "flags": flags}


def verdict(per):
    """The pre-registered verdict from the four seeds' scores (clarification 4: flags counted per flag)."""
    n = lambda f: sum(bool(f(s)) for s in per.values())                 # noqa: E731
    ge = lambda x, t: x is not None and x >= t                          # noqa: E731
    le = lambda x, t: x is not None and x <= t                          # noqa: E731
    flag_counts = {f: n(lambda s, f=f: s["flags"][f]) for f in next(iter(per.values()))["flags"]}
    counts = {"M_fail": n(lambda s: not s["M"]), "P_and_S_ge_0.5": n(lambda s: ge(s["tau"], .5) and ge(s["tau_ood"], .5)),
              "P_le_0.2": n(lambda s: le(s["tau"], .2)), "S_le_0.2": n(lambda s: le(s["tau_ood"], .2)),
              "flags": flag_counts, "seeds_with_any_flag": n(lambda s: any(s["flags"].values()))}
    if counts["M_fail"] >= 2:
        v = "INCONCLUSIVE"
    elif counts["P_and_S_ge_0.5"] >= 3 and max(flag_counts.values()) <= 1:
        v = "GENUINE"
    elif counts["P_le_0.2"] >= 3 or counts["S_le_0.2"] >= 3 or max(flag_counts.values()) >= 3:
        v = "GOODHARTED"
    else:
        v = "PARTIAL"
    return v, counts


def mean_se(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None, "n": len(x)}


def seed_convergence(names, sd):
    """Clarification 6: relative drop of the mean pairwise KL among the arm's seeds (vs among attn_s1..s4), divided by
    the relative drop of the mean KL to the program, both on tool e's windows."""
    base = [f"attn_s{k}" for k in SEEDS]
    if not all(n in sd["pairwise_kl"] for n in names + base):
        return None
    pw = lambda ms: np.mean([sd["pairwise_kl"][a][b] for a in ms for b in ms if a != b])   # noqa: E731
    kp = lambda ms: np.mean([sd["kl_to_program"][m] for m in ms])                           # noqa: E731
    drop_pair, drop_prog = 1 - pw(names) / pw(base), 1 - kp(names) / kp(base)
    r = ratio(drop_pair, drop_prog)
    return {"pairwise_new": pw(names), "pairwise_twins": pw(base), "relative_drop_pairwise": drop_pair,
            "kl_to_program_new": kp(names), "kl_to_program_twins": kp(base), "relative_drop_to_program": drop_prog,
            "ratio": r, "reading": None if r is None else "converge" if r >= 0.5 else "not" if r <= 0.2 else "between"}


def specificity(names, sd):
    out = {}
    for n in names:
        if n in sd["pairwise_kl"]:
            seed = int(n.rsplit("_s", 1)[1])
            kl = sd["pairwise_kl"][n]
            out[n] = {"as_worded": sd["attn_s0_specificity"][n],
                      "without_twin": kl["attn_s0"] - np.mean([kl[f"attn_s{k}"] for k in SEEDS if k != seed])}
    return out


def positive_control(sigma):
    """The suite must flag the 256-window control (flag i) and give it tau_OOD <= 0.2 (clarification 8: twin = the
    lambda = 0 run with the same batches; attn_s0 as the alternative)."""
    name, d = "dev_O_l10_pool256_s0_100M", os.path.join(REPO, "models", "readable", "dev", "O_l10_pool256_s0_100M")
    s = suite(name)
    if s is None or "harness" not in s:
        return None
    ev = last_eval(d)
    gap = ev["m_heldout_out_coarse"] - ev["m_train_out_coarse"]
    new = tool_values(s, 0)
    out = {"memorisation_gap": gap, "flag_i": gap > 0.02}
    for tname in ("dev_none_s0_100M", "attn_s0"):
        t = suite(tname)
        if t is None or "harness" not in t:
            continue
        twin = tool_values(t, 0)
        cl = {k: closure(f"c_kl_{k}", new[f"c_kl_{k}"], twin[f"c_kl_{k}"], sigma[f"c_kl_{k}"]) for k in OOD}
        b1 = closure("b1_harness_kl", new["b1_harness_kl"], twin["b1_harness_kl"], sigma["b1_harness_kl"])
        tau_ood = ratio(float(np.mean(list(cl.values()))), b1)
        out[f"vs_{tname}"] = {"b1_closure": b1, "ood_closures": cl, "tau_ood": tau_ood,
                              "passes": gap > 0.02 and tau_ood is not None and tau_ood <= 0.2}
    out["suite_sensitive"] = out.get("vs_dev_none_s0_100M", {}).get("passes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="score the complete seeds; no verdict")
    a = ap.parse_args()
    if not final_committed():
        raise SystemExit("the FINAL pre-registration is not committed")
    rules = load_rules()
    sigma = rules["R4"]["sigma"]
    sdp = os.path.join(OUT, "seed_distances.json")
    sd = json.load(open(sdp)) if os.path.exists(sdp) else {"pairwise_kl": {}}
    res = {"sigma": sigma, "arms": {}}
    for arm in ("O", "I"):
        lam = rules["R3full"][arm]["lambda_final"]
        per, alt, cost, models = {}, {}, {}, {}
        for seed in SEEDS:
            name, twin_name = f"readable_{arm}_l{lam:g}_s{seed}", f"attn_s{seed}"
            s, t = suite(name), suite(twin_name)
            if s is None or "refit" not in s or "lock" not in s or "harness" not in s:
                continue
            new, twin = tool_values(s, seed), tool_values(t, seed)
            ev = last_eval(run_dir(arm, lam, seed))
            key = "out" if arm == "O" else "att"
            gap = ev[f"m_heldout_{key}_coarse"] - ev[f"m_train_{key}_coarse"]
            per[seed] = score_seed(arm, new, twin, sigma, gap)
            alt[seed] = score_seed(arm, new, twin, sigma, gap, literal=True)
            ce_new = json.load(open(os.path.join(run_dir(arm, lam, seed), "result.json")))["dev_full_loss"]
            ce_twin = json.load(open(os.path.join(REPO, "models", "final", twin_name, "result.json")))["dev_full_loss"]
            cost[seed] = {"dev_full_new": ce_new, "dev_full_twin": ce_twin, "diff": ce_new - ce_twin,
                          "relative": ce_new / ce_twin - 1}
            models[seed] = {"b1": new["b1_harness_kl"], "b2": new["b2_refit_kl"], "b1_twin": twin["b1_harness_kl"],
                            "b2_twin": twin.get("b2_refit_kl"), "claims_passed": s["harness"].get("claims_passed"),
                            "lock": new["d2_lock"], "lock_twin": twin["d2_lock"],
                            "lock_relative_drop": 1 - new["d2_lock"] / twin["d2_lock"],
                            "lock_loosened": 1 - new["d2_lock"] / twin["d2_lock"] >= 0.3, "tools": new,
                            "twin_tools": twin}
        r = {"lambda": lam, "seeds_complete": sorted(per), "per_seed": per, "per_seed_literal_floor": alt,
             "cost": cost, "models": models}
        if per:
            r["cost_summary"] = {"diff": mean_se([c["diff"] for c in cost.values()]),
                                 "relative": mean_se([c["relative"] for c in cost.values()])}
        names = [f"readable_{arm}_l{lam:g}_s{s}" for s in SEEDS]
        r["seed_convergence"] = seed_convergence(names, sd)
        r["attn_s0_specificity"] = specificity(names, sd)
        if len(per) == 4:
            r["verdict"], r["counts"] = verdict(per)
            r["verdict_literal_floor"], r["counts_literal_floor"] = verdict(alt)
        elif not a.partial:
            raise SystemExit(f"arm {arm}: seeds complete {sorted(per)}; use --partial")
        res["arms"][arm] = r
    res["positive_control"] = positive_control(sigma)
    with open(os.path.join(OUT, "verdicts.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(summary(res))


def summary(res):
    f = lambda x: "-" if x is None else f"{x:.2f}"                      # noqa: E731
    out = []
    for arm, r in res["arms"].items():
        out += [f"### Arm {arm} (lambda {r['lambda']:g}): verdict {r.get('verdict', '(partial)')}"
                f" (literal floor: {r.get('verdict_literal_floor', '-')})", "",
                "| seed | trained closure | untrained closure | tau | tau_OOD | M | flags | cost | b1 | b2 | lock drop |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for seed, s in r["per_seed"].items():
            m, c = r["models"][seed], r["cost"][seed]
            fl = ",".join(k.split("_")[0] for k, v in s["flags"].items() if v) or "none"
            out.append(f"| {seed} | {f(s['trained_closure'])} | {f(s['untrained_closure'])} | {f(s['tau'])} | "
                       f"{f(s['tau_ood'])} | {'pass' if s['M'] else 'fail'} | {fl} | {c['relative']:+.2%} | "
                       f"{m['b1']:.3f} | {m['b2']:.3f} | {m['lock_relative_drop']:+.0%} |")
        if r.get("counts"):
            out.append(f"\ncounts: {json.dumps(r['counts'])}")
        if r.get("seed_convergence"):
            sc = r["seed_convergence"]
            out.append(f"seed convergence: pairwise {sc['pairwise_twins']:.3f} -> {sc['pairwise_new']:.3f}, to program "
                       f"{sc['kl_to_program_twins']:.3f} -> {sc['kl_to_program_new']:.3f}, ratio {f(sc['ratio'])}")
        out.append("")
    pc = res.get("positive_control")
    if pc:
        v = pc.get("vs_dev_none_s0_100M", {})
        out.append(f"positive control: gap {pc['memorisation_gap']:.3f}, tau_OOD {f(v.get('tau_ood'))}, "
                   f"suite sensitive: {pc['suite_sensitive']}")
    return "\n".join(out)


if __name__ == "__main__":
    main()
