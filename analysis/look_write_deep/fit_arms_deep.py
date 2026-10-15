"""Per model and head: base attention rule plus one pair-feature arm at a time, each fitted to the head's own pattern
(train_windows(32, 2024)); exact-model substitution at (layer, head) with later blocks recomputed, output KL on
train_windows(64, 2025); pure-pair correlation at d = 1 against the oracle table.

Arms (features from analysis/out/look_write_deep/<model>/bgu.npz, c1 at d = 1 and c2 at d = 2..7):
    own_<readout>   own BGU with readout ltotal | total | dv
    other_LxHy      another head's BGU (primary readout)
    shuffled        value-shuffled control (primary readout)
    echo / cgu      exploratory: model-echo part, and BGU minus it (primary readout)
    oqc             oracle ceiling: the head's own mean score s_ij - s_ii per (k, q, d), shrunk by N / (N + 20)

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.fit_arms_deep attn2L_s1 [--primary ltotal]
Writes analysis/out/look_write_deep/<model>/arms.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.look_write.fit_arms import Rule, double_centre, fit_rule, pattern_ce, wcorr
from analysis.look_write_deep import common_deep as C

MIN_COUNT = 20


def tables(z, primary):
    """Per-head slot tables: dict name -> (L, H, S) float64."""
    N = z["N"]
    bgu = lambda a: a / (N + C.SHRINK)                       # noqa: E731
    shuf = "lshuf" if primary == "ltotal" else "shuf"
    echo = "lecho" if primary == "ltotal" else "echo"
    t = {f"own_{r}": bgu(z[r]) for r in ("ltotal", "total", "dv")}
    t["shuffled"] = bgu(z[shuf])
    t["echo"] = bgu(z[echo])
    t["cgu"] = t[f"own_{primary}"] - t["echo"]
    t["oqc"] = np.where(N > 0, z["oqc"] / np.maximum(N, 1), 0.0) * (N / (N + C.SHRINK))
    return t


def slot_features(wins, keys, values):
    """values (S,) -> (F1, F2): F1[b, i, i-1] = value of slot (1, t_{i-1}, t_i); F2 the same for d = 2..7."""
    n, TT = wins.shape
    F1, F2 = np.zeros((n, TT, TT)), np.zeros((n, TT, TT))
    for d in range(1, C.D_MAX + 1):
        i = np.arange(d, TT)
        s = C.slot_of(keys, C.pair_key(d, wins[:, i - d], wins[:, i]))
        vals = np.where(s >= 0, values[np.maximum(s, 0)], 0.0)
        (F1 if d == 1 else F2)[:, i, i - d] = vals
    return F1, F2


def run(model_id, primary="ltotal", log=print):
    torch.set_num_threads(2)
    m = C.load(model_id)
    L, H, _, _ = C.dims(m)
    d = C.model_dir(model_id)
    z = np.load(os.path.join(d, "bgu.npz"))
    keys, N = z["keys"], z["N"]
    tabs = tables(z, primary)
    classes = json.load(open(os.path.join(d, "classes.json")))["heads"]
    shapes = C.token_shapes()
    fit_w = C.windows(C.FIT_SEED, C.N_FIT)[:, :C.T]
    ev_w = C.windows(C.EVAL_SEED, C.N_EVAL)[:, :C.T]
    A_fit, A_ev = C.patterns(m, fit_w), C.patterns(m, ev_w)
    lp_ref = C.logprobs(m, ev_w)
    rf, re_ = Rule(fit_w, shapes), Rule(ev_w, shapes)
    # pure-pair correlation at d = 1: slots with N >= 20
    sel = (keys // (C.V * C.V) == 1) & (N >= MIN_COUNT)
    kk, qq, ww = (keys[sel] // C.V) % C.V, keys[sel] % C.V, N[sel]

    def dc(vals):
        Y = np.zeros((C.V, C.V))
        Y[kk, qq] = vals
        return double_centre(Y, kk, qq, ww)

    res = {"model": model_id, "primary": primary, "n_blocks": L, "n_tokens": int(z["n_tokens"]), "heads": {}}
    for l in range(L):
        for h in range(H):
            t0 = time.time()
            cls = next(c for c in classes if c["layer"] == l and c["head"] == h)
            A_f = torch.as_tensor(A_fit[l][:, h].astype(np.float64))
            A_e = torch.as_tensor(A_ev[l][:, h].astype(np.float64))
            arms_spec = {name: tabs[name][l, h] for name in
                         ("own_ltotal", "own_total", "own_dv", "shuffled", "echo", "cgu", "oqc")}
            for l2 in range(L):
                for h2 in range(H):
                    if (l2, h2) != (l, h):
                        arms_spec[f"other_L{l2}H{h2}"] = tabs[f"own_{primary}"][l2, h2]
            out = {}
            for arm in ["base"] + list(arms_spec):
                if arm == "base":
                    ff, fe = {}, {}
                else:
                    F1f, F2f = slot_features(fit_w, keys, arms_spec[arm])
                    F1e, F2e = slot_features(ev_w, keys, arms_spec[arm])
                    ff = {"c1": torch.as_tensor(F1f), "c2": torch.as_tensor(F2f)}
                    fe = {"c1": torch.as_tensor(F1e), "c2": torch.as_tensor(F2e)}
                theta, loss, g = fit_rule(A_f, rf, ff)
                with torch.no_grad():
                    pat = torch.exp(re_.logpattern(theta, fe)).float().numpy()
                lp = C.logprobs(m, ev_w, sub=(l, h, pat))
                out[arm] = {"kl": C.kl(lp_ref, lp), "kl_win": C.kl_per_window(lp_ref, lp).tolist(),
                            "pattern_ce_fit": loss, "pattern_ce_eval": pattern_ce(A_e, re_, theta, fe),
                            "coef": {k: float(theta[k]) for k in ff}}
            base = out["base"]["kl"]
            for arm in out:
                out[arm]["gain"] = base - out[arm]["kl"]
            # pure-pair correlations at d = 1 (oracle table vs each head's primary BGU table)
            opp = dc(tabs["oqc"][l, h][sel])
            corr = {"own": wcorr(opp, dc(tabs[f"own_{primary}"][l, h][sel]), ww)}
            for l2 in range(L):
                for h2 in range(H):
                    if (l2, h2) != (l, h):
                        corr[f"other_L{l2}H{h2}"] = wcorr(opp, dc(tabs[f"own_{primary}"][l2, h2][sel]), ww)
            # near-duplicate heads: BGU tables at d = 1 correlated > 0.5 with the own table
            own1 = tabs[f"own_{primary}"][l, h][sel]
            dup = [f"other_L{l2}H{h2}" for l2 in range(L) for h2 in range(H) if (l2, h2) != (l, h)
                   and wcorr(own1, tabs[f"own_{primary}"][l2, h2][sel], ww) > 0.5]
            readout_corr = {r: wcorr(own1, tabs[f"own_{r}"][l, h][sel], ww) for r in ("ltotal", "total", "dv")}
            res["heads"][f"L{l}H{h}"] = {"class": cls, "arms": out, "pure_pair_corr": corr, "near_duplicates": dup,
                                         "readout_table_corr_d1": readout_corr}
            log(f"{model_id} L{l}H{h} ({'conf' if cls['confirmatory'] else 'expl'}): base {base:.4f}  "
                f"own {out['own_' + primary]['gain']:+.4f}  oqc {out['oqc']['gain']:+.4f}  "
                f"shuf {out['shuffled']['gain']:+.4f}  max other "
                f"{max(v['gain'] for k, v in out.items() if k.startswith('other')):+.4f}  "
                f"corr own {corr['own']:.3f}  [{time.time() - t0:.0f}s]")
    with open(os.path.join(d, f"arms_{primary}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--primary", default="ltotal", choices=["ltotal", "total"])
    a = ap.parse_args()
    for mid in a.models:
        run(mid, a.primary, log=lambda s: print(s, flush=True))
