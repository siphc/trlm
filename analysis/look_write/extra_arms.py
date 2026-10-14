"""EXPLORATORY (not pre-registered): alternative shapes of the exact token-token ceiling arm, to see how much P1
depends on the ceiling's form. Same base rule, fit and output test as fit_arms.py.

    tt_d17   : TT(t_i, t_j) only at distances 1..7, coefficients for d = 1 and d = 2..7 (the UOV arms' structure,
               and the structure of round 2's ceiling)
    tt_split : TT at all distances j < i with three coefficients (d = 1, d = 2..7, d >= 8)
    tt_self  : TT at all j <= i INCLUDING the self position (TT(t_i, t_i) at j = i), one coefficient: the full
               token-token term (the pre-registered ceiling leaves the self position out)

Writes analysis/out/look_write/<model>/arms_extra.json.
Run: OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write.extra_arms attn_s0 attn_s1 ...
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from analysis.look_write import common as C
from analysis.look_write.fit_arms import Rule, fit_rule, pattern_ce, tt_feature


def split(F, T, bands):
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    return [torch.as_tensor(F * ((dist >= lo) & (dist <= hi))[None]) for lo, hi in bands]


def run_model(model_id):
    C.set_threads()
    out_dir = C.model_dir(model_id)
    base_res = json.load(open(os.path.join(out_dir, "arms.json")))
    m = C.load(model_id)
    w = C.weights(m)
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    A_fit, A_ev = C.patterns(m, fit_w), C.patterns(m, ev_w)
    lp_ref = C.logprobs_sub(m, ev_w)
    rf, re_ = Rule(fit_w, shapes), Rule(ev_w, shapes)
    tt = C.tt_tables(w)
    res = {"model": model_id, "heads": {}}
    for h in range(C.H):
        t0 = time.time()
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        A_e = torch.as_tensor(A_ev[:, h].astype(np.float64))
        Ff, Fe = tt_feature(fit_w, tt[h]), tt_feature(ev_w, tt[h])
        ii = np.arange(C.T)
        Ff[:, ii, ii] = tt[h][fit_w, fit_w]          # self entries, used only by the (0, T) band of tt_self
        Fe[:, ii, ii] = tt[h][ev_w, ev_w]
        arms = {}
        for name, bands in (("tt_d17", [(1, 1), (2, 7)]), ("tt_split", [(1, 1), (2, 7), (8, C.T)]),
                            ("tt_self", [(0, C.T)])):
            names = ["c1", "c2", "c3"][:len(bands)]
            ff = dict(zip(names, split(Ff, C.T, bands)))
            fe = dict(zip(names, split(Fe, C.T, bands)))
            theta, loss, g = fit_rule(A_f, rf, ff)
            with torch.no_grad():
                pat = torch.exp(re_.logpattern(theta, fe)).float().numpy()
            klv = C.kl(lp_ref, C.logprobs_sub(m, ev_w, h, pat))
            base = base_res["heads"][str(h)]["arms"]["base"]["kl"]
            arms[name] = {"kl": klv, "gain": base - klv, "pattern_ce_fit": loss,
                          "pattern_ce_eval": pattern_ce(A_e, re_, theta, fe), "max_grad": g,
                          "coef": {k: float(theta[k]) for k in names}}
        res["heads"][h] = arms
        print(f"{model_id} head {h}: " + ", ".join(f"{k} gain {v['gain']:+.4f} coef "
                                                    f"{ {kk: round(vv, 3) for kk, vv in v['coef'].items()} }"
                                                    for k, v in arms.items()) + f" [{time.time() - t0:.0f}s]",
              flush=True)
    with open(os.path.join(out_dir, "arms_extra.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        run_model(mid)
