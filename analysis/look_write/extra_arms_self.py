"""EXPLORATORY (after addendum A2 failed): split the tt_self ceiling into its self-position part and its
earlier-token part, and ask whether own-vote usefulness covers the earlier-token part.

Arms (same base rule, fit and output test as fit_arms.py):
    self_only      TT(t_i, t_i) at j = i only (token-specific attention to the current token), one coefficient
    uov_self       own UOV (c1 at d = 1, c2 at d = 2..7) + TT at j = i (one coefficient)
    tt_self        reference: from arms_extra.json (TT at every j <= i)
If uov_self's gain approaches tt_self's, usefulness accounts for the earlier-token part of the token-token term and
the gap to the ceiling is the self-position part (where the vote's usefulness against the bigram is zero by
construction: E[next | q] = P1(. | q)).

Run: OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write.extra_arms_self attn_s1 attn_s2 attn_s3 attn_s4 attn_s0
Writes analysis/out/look_write/<model>/arms_self.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from analysis.look_write import common as C
from analysis.look_write.fit_arms import Rule, fit_rule, pair_features


def run_model(model_id):
    C.set_threads()
    m = C.load(model_id)
    w = C.weights(m)
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    out_dir = C.model_dir(model_id)
    base_res = json.load(open(os.path.join(out_dir, "arms.json")))
    extra = json.load(open(os.path.join(out_dir, "arms_extra.json")))
    A_fit, A_ev = C.patterns(m, fit_w), C.patterns(m, ev_w)
    lp_ref = C.logprobs_sub(m, ev_w)
    rf, re_ = Rule(fit_w, shapes), Rule(ev_w, shapes)
    tt = C.tt_tables(w)
    ii = np.arange(C.T)
    res = {"model": model_id, "heads": {}}
    for h in range(C.H):
        t0 = time.time()
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        uov = {d: np.load(os.path.join(out_dir, f"uov_h{h}_d{d}.npy")).astype(np.float64) for d in range(1, 8)}
        feats = {}
        for name, wins in (("fit", fit_w), ("ev", ev_w)):
            S = np.zeros((len(wins), C.T, C.T))
            S[:, ii, ii] = tt[h][wins, wins]
            F1, F2 = pair_features(wins, uov)
            feats[name] = {"self": torch.as_tensor(S), "c1": torch.as_tensor(F1), "c2": torch.as_tensor(F2)}
        arms = {}
        for arm, names in (("self_only", ["self"]), ("uov_self", ["c1", "c2", "self"])):
            ff = {k: feats["fit"][k] for k in names}
            fe = {k: feats["ev"][k] for k in names}
            theta, loss, g = fit_rule(A_f, rf, ff)
            with torch.no_grad():
                pat = torch.exp(re_.logpattern(theta, fe)).float().numpy()
            klv = C.kl(lp_ref, C.logprobs_sub(m, ev_w, h, pat))
            base = base_res["heads"][str(h)]["arms"]["base"]["kl"]
            arms[arm] = {"kl": klv, "gain": base - klv, "coef": {k: float(theta[k]) for k in names}}
        arms["tt_self"] = {"gain": extra["heads"][str(h)]["tt_self"]["gain"]}
        res["heads"][h] = arms
        print(f"{model_id} head {h}: self_only {arms['self_only']['gain']:+.4f}, uov_self {arms['uov_self']['gain']:+.4f}, "
              f"tt_self {arms['tt_self']['gain']:+.4f} [{time.time() - t0:.0f}s]", flush=True)
    with open(os.path.join(out_dir, "arms_self.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        run_model(mid)
