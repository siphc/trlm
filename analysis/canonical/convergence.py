"""Study C part B measurements (logs/study_C_readable/prereg_canonical_addendum.md): how much of the variation between training runs a
shared target removes, per unit of cross-entropy cost, for the readable target (program v1.3; arms O and I) and the
non-readable one (attn_s0; controls Octl and Ictl), in behaviour and in mechanism.

Per treatment X (four models, seeds 1-4, each against its untreated twin attn_s{k}):
    D       mean pairwise KL among the four models (ordered pairs; seed_distances.json, train_windows(64, 9100))
    R       1 - D_X / D_twins                                  behavioural convergence
    C       mean relative dev_full cross-entropy cost against the twins (result.json)
    E       R / C                                              convergence per unit of cost
    ratio   relative drop of D / relative drop of the mean KL to X's own target (program or attn_s0)
    b1      harness KL to program v1.3 (suite.json), against the twins'
Mechanism, on the head-aligned checkpoints (roles LOCAL-0, RECALL, LOCAL-2 on heads 0, 1, 2; suite.py align):
    P_h     mean pairwise attention-pattern KL of role h between the four models (train_windows(16, 9125), rows i >= 1)
    Q_h     mean pairwise value-code subspace overlap of role h (named_features.model_codes, uniform token weights,
            named_features_more.overlap)
    R_P     1 - mean_h P_h / the twins'                       R_Q   (Q - Q_twins) / (1 - Q_twins)
Uncertainty: leave-one-seed-out jackknife over the four seeds (each statistic recomputed on three seeds and their three
twins); for the paired comparisons arm vs control, the jackknife of log(E_arm / E_control).

    .venv/bin/python -m analysis.canonical.convergence [--treatments O_l0.3,I_l0.3,O_l1,Octl_l<x>,Ictl_l<y>]
                                                      [--compare O_l0.3:Octl_l<x>,I_l0.3:Ictl_l<y>] [--no_mech]
Writes analysis/out/canonical/convergence.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from analysis.readable_training.common import OUT as RT_OUT
from analysis.readable_training.common import REPO

OUT = os.path.join(REPO, "analysis", "out", "canonical")
SEEDS = (1, 2, 3, 4)
ROLES = (0, 1, 2)
# pre-registered thresholds (logs/study_C_readable/prereg_canonical_addendum.md, "Decision rules")
EFF_RATIO = 1.5            # one target counts as more efficient only if E differs by this factor ...
Z_MIN = 2.0                # ... and by at least 2 jackknife standard errors of the log ratio
C_MIN = 0.0025             # below a mean relative cost of 0.25% efficiency is not computed (cost is noise-level)
DR_MIN = 0.05              # primary comparison at matched cost: a convergence difference of at least 0.05 ...
MATCH_C = 0.004            # ... counts only if the two mean costs differ by at most 0.4 percentage points;
                           # otherwise the efficiency ratio E decides (EFF_RATIO, Z_MIN)


def name(tr, k):
    return f"readable_{tr}_s{k}"


def model_dir(tr, k):
    return os.path.join(REPO, "models", "readable", f"{tr}_s{k}")


def target_of(tr):
    return "attn_s0" if tr.split("_")[0].endswith("ctl") else "program"


def ce(path):
    return json.load(open(os.path.join(path, "result.json")))["dev_full_loss"]


def suite(n):
    return json.load(open(os.path.join(RT_OUT, n, "suite.json")))


def mean_pairwise(pw, names):
    return float(np.mean([pw[a][b] for a, b in itertools.permutations(names, 2)]))


def behaviour(tr, seeds, sd):
    pw = sd["pairwise_kl"]
    new, tw = [name(tr, k) for k in seeds], [f"attn_s{k}" for k in seeds]
    d_new, d_tw = mean_pairwise(pw, new), mean_pairwise(pw, tw)
    cost = [ce(model_dir(tr, k)) / ce(os.path.join(REPO, "models", "final", f"attn_s{k}")) - 1 for k in seeds]
    if target_of(tr) == "program":
        t_new = np.mean([sd["kl_to_program"][n] for n in new])
        t_tw = np.mean([sd["kl_to_program"][n] for n in tw])
    else:
        t_new = np.mean([pw[n]["attn_s0"] for n in new])
        t_tw = np.mean([pw[n]["attn_s0"] for n in tw])
    R = 1 - d_new / d_tw
    C = float(np.mean(cost))
    drop_t = 1 - t_new / t_tw
    return {"D": d_new, "D_twins": d_tw, "R": R, "C": C, "cost_per_seed": cost,
            "E": R / C if C >= C_MIN else None, "kl_to_target": float(t_new), "kl_to_target_twins": float(t_tw),
            "convergence_ratio": R / drop_t if abs(drop_t) > 1e-9 else None}


def jackknife(f, seeds=SEEDS):
    """f(seeds) -> float or None; returns (value on all seeds, jackknife SE) or (value, None)."""
    full = f(seeds)
    loo = [f(tuple(s for s in seeds if s != k)) for k in seeds]
    if full is None or any(v is None for v in loo):
        return full, None
    n = len(seeds)
    loo = np.asarray(loo, np.float64)
    return full, float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))


def mech_cache(paths, W):
    """Per model: attention of roles 0-2 on W and the bases of their value codes."""
    from analysis.attention_fit import model_attention
    from analysis.token_geometry.named_features import V, basis, model_codes

    w = np.full(V, 1.0 / V)
    out = {}
    for n, p in paths.items():
        A = model_attention(p, W)[:, list(ROLES)].astype(np.float64)
        codes = model_codes(p)[1]
        out[n] = {"att": A, "Q": [basis(codes[h], w) for h in ROLES]}
    return out


def pattern_kl(A, B):
    a, b = A[:, :, 1:], B[:, :, 1:]                                   # (n, T-1, T) rows i >= 1
    return float((a * (np.log(np.maximum(a, 1e-30)) - np.log(np.maximum(b, 1e-30)))).sum(-1).mean())


def mechanism(names_new, names_tw, cache):
    from analysis.token_geometry.named_features_more import overlap

    def stats(names):
        P = {h: np.mean([pattern_kl(cache[a]["att"][:, i], cache[b]["att"][:, i])
                         for a, b in itertools.permutations(names, 2)]) for i, h in enumerate(ROLES)}
        Q = {h: np.mean([overlap(cache[b]["Q"][i], cache[a]["Q"][i]) for a, b in itertools.permutations(names, 2)])
             for i, h in enumerate(ROLES)}
        return P, Q

    Pn, Qn = stats(names_new)
    Pt, Qt = stats(names_tw)
    pn, pt = np.mean(list(Pn.values())), np.mean(list(Pt.values()))
    qn, qt = np.mean(list(Qn.values())), np.mean(list(Qt.values()))
    return {"P": {str(h): float(v) for h, v in Pn.items()}, "P_twins": {str(h): float(v) for h, v in Pt.items()},
            "Q": {str(h): float(v) for h, v in Qn.items()}, "Q_twins": {str(h): float(v) for h, v in Qt.items()},
            "R_P": float(1 - pn / pt), "R_Q": float((qn - qt) / (1 - qt))}


def classify_matched(dr, z):
    if dr >= DR_MIN and z >= Z_MIN:
        return "readable target removes more"
    if dr <= -DR_MIN and z <= -Z_MIN:
        return "non-readable target removes more"
    return "no difference shown"


def classify_efficiency(e_arm, e_ctl, z):
    if e_arm is None or e_ctl is None or z is None:
        return "not computed (a cost below 0.25%)"
    r = e_arm / e_ctl if e_ctl > 0 else float("inf")
    if r >= EFF_RATIO and z >= Z_MIN:
        return "readable target more efficient"
    if r <= 1 / EFF_RATIO and z <= -Z_MIN:
        return "non-readable target more efficient"
    return "no difference shown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatments", default="O_l0.3,I_l0.3")
    ap.add_argument("--compare", default="", help="arm:control pairs, comma-separated")
    ap.add_argument("--no_mech", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "convergence.json"))
    a = ap.parse_args()
    trs = a.treatments.split(",")
    sd = json.load(open(os.path.join(RT_OUT, "seed_distances.json")))
    res = {"treatments": {}, "comparisons": {}, "thresholds": {"EFF_RATIO": EFF_RATIO, "Z_MIN": Z_MIN, "C_MIN": C_MIN}}
    cache = None
    if not a.no_mech:
        from analysis.fitlib import train_windows

        W = train_windows(16, 9125)
        paths = {f"attn_s{k}": os.path.join(RT_OUT, f"attn_s{k}", "aligned.pt") for k in SEEDS}
        paths.update({name(t, k): os.path.join(RT_OUT, name(t, k), "aligned.pt") for t in trs for k in SEEDS})
        cache = mech_cache(paths, W)
    for tr in trs:
        r = behaviour(tr, SEEDS, sd)
        for key in ("R", "C", "E"):
            r[f"{key}_se"] = jackknife(lambda s, key=key: behaviour(tr, s, sd)[key])[1]
        r["b1"] = float(np.mean([suite(name(tr, k))["harness"]["kl"] for k in SEEDS]))
        r["b1_twins"] = float(np.mean([suite(f"attn_s{k}")["harness"]["kl"] for k in SEEDS]))
        if cache is not None:
            r["mechanism"] = mechanism([name(tr, k) for k in SEEDS], [f"attn_s{k}" for k in SEEDS], cache)
            for key in ("R_P", "R_Q"):
                r["mechanism"][f"{key}_se"] = jackknife(lambda s, key=key: mechanism(
                    [name(tr, k) for k in s], [f"attn_s{k}" for k in s], cache)[key])[1]
        res["treatments"][tr] = r
        print(tr, {k: v for k, v in r.items() if k not in ("cost_per_seed", "mechanism")}, flush=True)
    for pair in (p for p in a.compare.split(",") if p):
        arm, ctl = pair.split(":")

        def logratio(s):
            ea, ec = behaviour(arm, s, sd)["E"], behaviour(ctl, s, sd)["E"]
            return None if ea is None or ec is None or ea <= 0 or ec <= 0 else float(np.log(ea / ec))

        lr, se = jackknife(logratio)
        z = lr / se if lr is not None and se else None
        dr, dr_se = jackknife(lambda s: behaviour(arm, s, sd)["R"] - behaviour(ctl, s, sd)["R"])
        dz = dr / dr_se if dr_se else None
        ta, tc = res["treatments"][arm], res["treatments"][ctl]
        ea, ec = ta["E"], tc["E"]
        dc = ta["C"] - tc["C"]
        matched = abs(dc) <= MATCH_C
        c = {"matched_cost": matched, "delta_C": dc, "delta_R": dr, "delta_R_se": dr_se, "delta_R_z": dz,
             "E_arm": ea, "E_control": ec, "log_ratio": lr, "log_ratio_se": se, "z": z,
             "reading": classify_matched(dr, dz if dz is not None else 0.0) if matched
             else classify_efficiency(ea, ec, z)}
        if "mechanism" in ta and "mechanism" in tc:
            for key in ("R_P", "R_Q"):
                c[f"delta_{key}"] = ta["mechanism"][key] - tc["mechanism"][key]
        res["comparisons"][pair] = c
        print(pair, res["comparisons"][pair], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
