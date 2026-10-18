"""Study A pipeline checks (logs/prereg_fixed_votes.md, "Pipeline checks before any confirmatory analysis").

  (a) goal 1's fitting code still reproduces its stored results: attn_s1 head 0, arms base, own and tt, against
      analysis/out/look_write/attn_s1/arms.json (within 0.002 nats);
  (b) synthetic: in a fixed-vote model (beta of the head set to 1), head 3's pattern is replaced by
      softmax(base rule + c x own usefulness) with known parameters; the own arm must recover each c within 10% and
      gain at least 0.9 x the base arm's KL.

    OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.fixed_votes.check_pipeline <fixed-vote checkpoint>
Writes analysis/out/fixed_votes/pipeline_checks.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from analysis.fixed_votes import law as FL
from analysis.look_write import common as C
from analysis.look_write import fit_arms as FA

C_TRUE = {"c1": 1.0, "c2": 0.6, "cb": 0.3}
BETA_TRUE = [1.5, 0.8, 0.4, 0.0, -0.4, -1.0, -2.0, -3.0, -4.0]        # distance buckets 1, 2, 3, 4, 5-7, ..., 64+
SK_TRUE = [1.0, 0.5, 2.0, 3.0, 0.0, -0.3, 0.2, 0.0]                    # self x query shape, key shape


def check_goal1(h=0, tol=0.002):
    m = C.load("attn_s1")
    w = C.weights(m)
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    A_fit, A_ev = C.patterns(m, fit_w), C.patterns(m, ev_w)
    lp_ref = C.logprobs_sub(m, ev_w)
    rf, re = FA.Rule(fit_w, shapes), FA.Rule(ev_w, shapes)
    tabs = FA.load_tables("attn_s1", "uov", h)
    f, e = FA.pair_features(fit_w, tabs), FA.pair_features(ev_w, tabs)
    tt = C.tt_tables(w)[h]
    arms = {"base": ({}, {}),
            "own": ({"c1": torch.as_tensor(f[0]), "c2": torch.as_tensor(f[1])},
                    {"c1": torch.as_tensor(e[0]), "c2": torch.as_tensor(e[1])}),
            "tt": ({"c": torch.as_tensor(FA.tt_feature(fit_w, tt))}, {"c": torch.as_tensor(FA.tt_feature(ev_w, tt))})}
    stored = json.load(open(os.path.join(C.model_dir("attn_s1"), "arms.json")))["heads"][str(h)]["arms"]
    A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
    out = {}
    for name, (ff, fe) in arms.items():
        theta, _, _ = FA.fit_rule(A_f, rf, ff)
        with torch.no_grad():
            pat = torch.exp(re.logpattern(theta, fe)).float().numpy()
        kl = C.kl(lp_ref, C.logprobs_sub(m, ev_w, h, pat))
        out[name] = {"kl": kl, "stored": stored[name]["kl"], "diff": kl - stored[name]["kl"]}
    out["pass"] = all(abs(v["diff"]) <= tol for v in out.values() if isinstance(v, dict))
    return out


def check_synthetic(path, h=3):
    from src.train.train import load_model

    m = load_model(path)
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        m.beta[h] = 1.0
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    rf, re = FA.Rule(fit_w, shapes), FA.Rule(ev_w, shapes)
    own = m.cfg.names[h]
    ff, fe = FL.u_features(fit_w, own, 1.0), FL.u_features(ev_w, own, 1.0)
    true = {"beta": torch.tensor(BETA_TRUE, dtype=torch.float64), "sk": torch.tensor(SK_TRUE, dtype=torch.float64),
            **{k: torch.tensor(v, dtype=torch.float64) for k, v in C_TRUE.items()}}
    with torch.no_grad():
        A_f = torch.exp(rf.logpattern(true, ff))
        A_e = torch.exp(re.logpattern(true, fe)).numpy()
    lp_ref = FL.logprobs(m, ev_w, h, A_e)                       # the "model" is the one with the hand-made pattern
    out = {"table": own, "c_true": C_TRUE, "arms": {}}
    for name, (f1, f2) in {"base": ({}, {}), "own": (ff, fe)}.items():
        theta, _, _ = FA.fit_rule(A_f, rf, f1)
        with torch.no_grad():
            pat = torch.exp(re.logpattern(theta, f2)).numpy()
        out["arms"][name] = {"kl": C.kl(lp_ref, FL.logprobs(m, ev_w, h, pat)),
                             "coef": {k: float(v) for k, v in theta.items() if k not in ("beta", "sk")}}
    base, own_kl = out["arms"]["base"]["kl"], out["arms"]["own"]["kl"]
    out["gain_over_base_kl"] = (base - own_kl) / base
    out["coef_rel_err"] = {k: abs(out["arms"]["own"]["coef"][k] / v - 1) for k, v in C_TRUE.items()}
    out["pass"] = out["gain_over_base_kl"] >= 0.9 and max(out["coef_rel_err"].values()) <= 0.10
    return out


def main():
    C.set_threads()
    res = {"goal1_reproduction": check_goal1()}
    print(json.dumps(res["goal1_reproduction"]), flush=True)
    res["synthetic"] = check_synthetic(sys.argv[1])
    print(json.dumps(res["synthetic"]), flush=True)
    os.makedirs(FL.OUT, exist_ok=True)
    with open(os.path.join(FL.OUT, "pipeline_checks.json"), "w") as f:
        json.dump({"model_for_synthetic": sys.argv[1], **res}, f, indent=1)


if __name__ == "__main__":
    main()
