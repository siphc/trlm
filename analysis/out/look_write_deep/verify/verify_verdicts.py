"""Independent verifier: D1-D5 per confirmatory head and family verdicts, written from the text of
logs/prereg_deep_look_write.md only (not from summarize_deep.py).

Inputs: analysis/out/look_write_deep/<model>/{arms_ltotal.json, classes.json}, rules.json.
Optional independent recomputation from bgu.npz (--tables): near-duplicate lists, primary-vs-DV table correlation at
d = 1, pure-pair correlations.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_verdicts.py [--tables]
Writes analysis/out/look_write_deep/verify/verdicts_verify.json
"""
import json
import os
import sys

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
OUT = os.path.join(REPO, "analysis", "out", "look_write_deep")
MODELS = [f"attn2L_s{s}" for s in range(1, 5)] + [f"attn4L_s{s}" for s in range(1, 5)]
V = 2000


def paired_se(kw_a, kw_b):
    """SE of mean(kw_a - kw_b) over windows (paired)."""
    d = np.asarray(kw_a, float) - np.asarray(kw_b, float)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def family(layer, n_blocks):
    if layer == 0:
        return "F1"
    if layer == n_blocks - 1:
        return "F2"
    if n_blocks == 4 and layer in (1, 2):
        return "F3"
    return None


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def double_centre_resid(y, k, q, w, iters=500):
    """Weighted additive fit y ~ a_k + b_q by backfitting (alternating weighted means); returns residual.
    Different algorithm from the study's preconditioned CG."""
    a = np.zeros(V)
    b = np.zeros(V)
    wk = np.bincount(k, w, V)
    wq = np.bincount(q, w, V)
    prev = None
    for it in range(iters):
        r = y - b[q]
        a = np.where(wk > 0, np.bincount(k, w * r, V) / np.maximum(wk, 1e-300), 0.0)
        r = y - a[k]
        b = np.where(wq > 0, np.bincount(q, w * r, V) / np.maximum(wq, 1e-300), 0.0)
        res = y - a[k] - b[q]
        if prev is not None and np.max(np.abs(res - prev)) < 1e-12 * (np.abs(y).max() + 1e-300):
            break
        prev = res
    return res, it


def table_checks(model, arms):
    """Recompute near-duplicates, primary-vs-DV correlation and pure-pair correlations from bgu.npz."""
    z = np.load(os.path.join(OUT, model, "bgu.npz"))
    keys, N = z["keys"], z["N"]
    d = keys // (V * V)
    sel = (d == 1) & (N >= 20)
    k = ((keys // V) % V)[sel]
    q = (keys % V)[sel]
    w = N[sel]
    lt = z["ltotal"][:, :, sel] / (N[sel] + 20.0)
    dv = z["dv"][:, :, sel] / (N[sel] + 20.0)
    oqc = z["oqc"][:, :, sel] / N[sel] * (N[sel] / (N[sel] + 20.0))      # mean, shrunk by N/(N+20)
    L, H = lt.shape[:2]
    resid_cache = {}

    def dc(name, y):
        if name not in resid_cache:
            resid_cache[name] = double_centre_resid(y, k, q, w)[0]
        return resid_cache[name]

    out = {}
    for hk, rec in arms["heads"].items():
        l, h = int(hk[1]), int(hk[3])
        own = lt[l, h]
        r = {"n_pairs": int(sel.sum())}
        r["dup_weighted"] = sorted(f"other_L{l2}H{h2}" for l2 in range(L) for h2 in range(H)
                                   if (l2, h2) != (l, h) and wcorr(own, lt[l2, h2], w) > 0.5)
        r["dup_unweighted"] = sorted(f"other_L{l2}H{h2}" for l2 in range(L) for h2 in range(H)
                                     if (l2, h2) != (l, h) and wcorr(own, lt[l2, h2], np.ones_like(w)) > 0.5)
        r["max_other_tablecorr_w"] = max(wcorr(own, lt[l2, h2], w) for l2 in range(L) for h2 in range(H)
                                         if (l2, h2) != (l, h))
        r["dv_corr_w"] = wcorr(own, dv[l, h], w)
        r["dv_corr_unw"] = wcorr(own, dv[l, h], np.ones_like(w))
        tgt = dc(f"oqc{l}{h}", oqc[l, h])
        # convention A: both double-centred; convention B: target double-centred, raw BGU table
        r["pp_own_dc"] = wcorr(tgt, dc(f"lt{l}{h}", own), w)
        r["pp_own_raw"] = wcorr(tgt, own, w)
        r["pp_other_dc"] = {f"other_L{l2}H{h2}": wcorr(tgt, dc(f"lt{l2}{h2}", lt[l2, h2]), w)
                            for l2 in range(L) for h2 in range(H) if (l2, h2) != (l, h)}
        out[hk] = r
    return out


def main(do_tables):
    rules = json.load(open(os.path.join(OUT, "rules.json")))
    primary = rules["R1"]["primary"]
    theta = rules["R2"]["theta"]
    c1r, c2r = rules["R3"]["c1_median"], rules["R3"]["c2_median"]
    heads = {}
    tchecks = {}
    for m in MODELS:
        arms = json.load(open(os.path.join(OUT, m, f"arms_{primary}.json")))
        assert arms["primary"] == primary, (m, arms["primary"])
        classes = json.load(open(os.path.join(OUT, m, "classes.json")))
        L = classes["n_blocks"]
        assert arms["n_blocks"] == L
        if do_tables:
            tchecks[m] = table_checks(m, arms)
        for c in classes["heads"]:
            # re-derive the confirmatory flag from the fixed rules
            conf = (c["mean_distance"] <= 10 and not c["prev_attention"] >= 0.5 and not c["induction_score"] >= 0.3
                    and c["earlier_mass"] >= 0.15)
            assert conf == c["confirmatory"], (m, c)
            if not conf:
                continue
            hk = f"L{c['layer']}H{c['head']}"
            rec = arms["heads"][hk]
            A = rec["arms"]
            base = A["base"]
            own = A[f"own_{primary}"]
            dv = A["own_dv"]
            oqc = A["oqc"]
            shuf = A["shuffled"]
            # gains from KLs (and check stored gains)
            g = {a: base["kl"] - A[a]["kl"] for a in A}
            for a in A:
                assert abs(g[a] - A[a]["gain"]) < 1e-12
                assert abs(np.mean(A[a]["kl_win"]) - A[a]["kl"]) < 1e-9, (m, hk, a)
            se_oqc = paired_se(base["kl_win"], oqc["kl_win"])
            se_own = paired_se(base["kl_win"], own["kl_win"])
            floor = max(0.01, 3 * se_oqc)
            ratio_eval = g["oqc"] >= floor
            if ratio_eval:
                d1 = g[f"own_{primary}"] >= theta * g["oqc"]
            else:
                d1 = g[f"own_{primary}"] >= 0.15 * base["kl"]
            dups = rec["near_duplicates"]
            others = {a: g[a] for a in A if a.startswith("other_") and a not in dups}
            max_ctrl = max(list(others.values()) + [g["shuffled"]])
            d2 = (g[f"own_{primary}"] > 0) and (max_ctrl <= 0.25 * g[f"own_{primary}"])
            pp = rec["pure_pair_corr"]
            pp_others = {a: v for a, v in pp.items() if a != "own" and a not in dups}
            d3 = pp["own"] >= 0.25 and all(pp["own"] > v for v in pp_others.values())
            # D4 head-level parts: primary vs DV, paired SE of the difference
            se_pd = paired_se(dv["kl_win"], own["kl_win"])                 # gain_primary - gain_dv = kl_dv - kl_own
            diff = g[f"own_{primary}"] - g["own_dv"]
            tcorr = rec["readout_table_corr_d1"]["dv"]
            c1, c2 = own["coef"]["c1"], own["coef"]["c2"]
            d5 = (c1r / 1.5 <= c1 <= c1r * 1.5) and (c2r / 1.5 <= c2 <= c2r * 1.5)
            heads[f"{m}/{hk}"] = {
                "model": m, "layer": c["layer"], "head": c["head"], "family": family(c["layer"], L),
                "base_kl": base["kl"], "own_gain": g[f"own_{primary}"], "oracle_gain": g["oqc"],
                "se_oracle": se_oqc, "se_own": se_own, "floor": floor, "ratio_evaluable": bool(ratio_eval),
                "own_over_oracle": g[f"own_{primary}"] / g["oqc"] if g["oqc"] != 0 else None,
                "own_over_base": g[f"own_{primary}"] / base["kl"],
                "D1": bool(d1), "max_control": max_ctrl, "shuffled_gain": g["shuffled"],
                "max_other_gain": max(others.values()) if others else None,
                "max_other_name": max(others, key=others.get) if others else None,
                "near_duplicates": dups, "D2": bool(d2),
                "pp_own": pp["own"], "pp_max_other": max(pp_others.values()) if pp_others else None,
                "D3": bool(d3), "D123": bool(d1 and d2 and d3),
                "dv_gain": g["own_dv"], "primary_minus_dv": diff, "se_primary_minus_dv": se_pd,
                "D4_not_worse": bool(diff >= -2 * se_pd), "D4_better": bool(diff >= 2 * se_pd),
                "table_corr_dv": tcorr, "tables_differ": bool(tcorr < 0.5),
                "c1": c1, "c2": c2, "D5": bool(d5),
            }
    fams = {}
    for f in ("F1", "F2", "F3"):
        hs = [v for v in heads.values() if v["family"] == f]
        n = len(hs)
        n123 = sum(v["D123"] for v in hs)
        nd1fail = sum(not v["D1"] for v in hs)
        if n < 4:
            verdict = "INCONCLUSIVE"
        elif n123 >= 0.8 * n:
            verdict = "SUPPORTED"
        elif nd1fail >= 0.5 * n:
            verdict = "REFUTED"
        else:
            verdict = "MIXED"
        rec = {"n": n, "n_D1": sum(v["D1"] for v in hs), "n_D2": sum(v["D2"] for v in hs),
               "n_D3": sum(v["D3"] for v in hs), "n_D123": n123, "n_D1_fail": nd1fail,
               "n_ratio_evaluable": sum(v["ratio_evaluable"] for v in hs), "n_D5": sum(v["D5"] for v in hs),
               "verdict": verdict}
        if f in ("F1", "F3"):
            not_worse = all(v["D4_not_worse"] for v in hs)
            differ = [v for v in hs if v["tables_differ"]]
            better = sum(v["D4_better"] for v in differ)
            rec.update({"D4_all_not_worse": not_worse, "n_not_worse": sum(v["D4_not_worse"] for v in hs),
                        "n_tables_differ": len(differ), "n_better_among_differ": better,
                        "D4": bool(not_worse and (len(differ) == 0 or better >= 0.75 * len(differ))),
                        "D4_if_vacuous_fails": bool(not_worse and len(differ) > 0 and better >= 0.75 * len(differ))})
        fams[f] = rec
    res = {"primary": primary, "theta": theta, "c1_ref": c1r, "c2_ref": c2r, "families": fams,
           "headline": fams["F1"]["verdict"] == "SUPPORTED" and fams["F2"]["verdict"] == "SUPPORTED",
           "heads": heads}
    if do_tables:
        res["table_checks"] = tchecks
    with open(os.path.join(OUT, "verify", "verdicts_verify.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    for f, r in fams.items():
        print(f, r)
    print("headline", res["headline"])


if __name__ == "__main__":
    main("--tables" in sys.argv)
