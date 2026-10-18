"""Study A measurements on one fixed-vote model (logs/prereg_fixed_votes.md): M1 vote strengths and vote-removal
shifts (R4), M3 law fit and output test, M4 pure-pair correlation, M5 distance profile, M6 coupling.

    OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.fixed_votes.law models/fixed_votes/base_s1/final.pt

The fitting code is goal 1's (analysis/look_write/fit_arms.py: base rule, L-BFGS fit, pair features, double-centred
pure-pair correlation), with the same windows. Usefulness features are beta_h * U_T (analysis/fixed_votes/usefulness.py),
so fitted coefficients are attention score per logit of usefulness. Arms per head h with table T_h:
  base               the base rule only (9 distance buckets, self x query shape, key shape)
  own                + own usefulness: c1 (d = 1), c2 (d = 2..7), cb (band d = 8..63)
  other_<T>          + the usefulness of another head's table T (the other heads of this model), same three features
  alt_<T>            + a table that no head of this model has (head 3's crossover test, skip2 vs skip3)
  perm               + the usefulness of T_h with its rows permuted (token-shuffled control)
  tt                 + the exact token-token score TT_h(q, k) at keys before the query (goal 1's pre-registered ceiling)
  tt_self            + TT_h at every key including the query (goal 1's addendum ceiling)
Writes analysis/out/fixed_votes/<name>/law.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from analysis.look_write import common as C
from analysis.look_write.fit_arms import (Rule, band_feature, double_centre, entropy, fit_rule, pair_features,
                                          pattern_ce, tt_feature, wcorr)
from analysis.fixed_votes.usefulness import BAND, OUT as U_DIR
from src.train.fixed_votes import TABLE_NAMES

OUT = os.path.join(C.REPO, "analysis", "out", "fixed_votes")


def name_of(path):
    return C.out_name(path).replace("fixed_votes_", "", 1)


def fv_weights(m):
    """The attention weights in analysis/look_write/common.weights' layout (for tt_tables and pure_pair_scores)."""
    H, Dh = m.cfg.n_heads, m.cfg.d_head
    W = m.qk.weight.detach().double().numpy()
    return {"wte": m.wte.weight.detach().double().numpy(), "wpe": m.wpe.weight.detach().double().numpy(),
            "g1": m.ln.weight.detach().double().numpy(),
            "Wq": [W[h * Dh:(h + 1) * Dh].T for h in range(H)],
            "Wk": [W[H * Dh + h * Dh:H * Dh + (h + 1) * Dh].T for h in range(H)]}


def patterns(m, wins, bs=8):
    with torch.no_grad():
        return np.concatenate([m.patterns(torch.as_tensor(np.asarray(wins[i:i + bs], np.int64))).numpy()
                               for i in range(0, len(wins), bs)])


def logprobs(m, wins, head=None, pats=None, bs=8):
    out = []
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs], np.int64))
            ov = None if head is None else {head: torch.as_tensor(pats[i:i + bs])}
            out.append(torch.log_softmax(m(x, override=ov).double(), -1).float().numpy())
    return np.concatenate(out)


def usefulness_tables(table, prefix="U"):
    tabs = {d: np.load(os.path.join(U_DIR, f"{prefix}_{table}_d{d}.npy")) for d in range(1, C.D_MAX + 1)}
    return tabs, np.load(os.path.join(U_DIR, f"{prefix}_{table}_band.npy"))


def u_features(wins, table, scale, prefix="U"):
    tabs, band = usefulness_tables(table, prefix)
    f1, f2 = pair_features(wins, {d: scale * t for d, t in tabs.items()})
    fb = band_feature(wins, scale * band, *BAND)
    return {"c1": torch.as_tensor(f1), "c2": torch.as_tensor(f2), "cb": torch.as_tensor(fb)}


def tt_self_feature(wins, tt):
    n, T = wins.shape
    F = tt[wins[:, :, None], wins[:, None, :]].astype(np.float64)
    return F * np.tril(np.ones((T, T)))[None]


def dev_ce(m, beta_off=None, bs=25):
    """dev_fast cross-entropy, optionally with one head's vote removed (beta_h = 0, attention untouched)."""
    from src.train.data import load_eval

    wins = load_eval("dev_fast")
    keep = m.beta.detach().clone()
    if beta_off is not None:
        with torch.no_grad():
            m.beta[beta_off] = 0.0
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(wins[i:i + bs].astype(np.int64))
            lp = torch.log_softmax(m(x[:, :-1]).double(), -1)
            tot += float(-lp.gather(-1, x[:, 1:, None]).sum())
            n += x[:, 1:].numel()
    with torch.no_grad():
        m.beta.copy_(keep)
    return tot / n


def run(path, log=print):
    from src.train.train import load_model

    C.set_threads()
    m = load_model(path)
    for p in m.parameters():
        p.requires_grad_(False)
    names, H = m.cfg.names, m.cfg.n_heads
    beta = [float(b) for b in m.beta]
    out_dir = os.path.join(OUT, name_of(path))
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": path, "tables": names, "beta": beta, "beta_follow": float(m.beta_follow)}
    # M1: vote removal (R4)
    ce = dev_ce(m)
    res["dev_fast_ce"] = ce
    res["vote_removal_shift"] = {h: dev_ce(m, h) - ce for h in range(H)}
    res["active"] = {h: res["vote_removal_shift"][h] >= 0.01 for h in range(H)}
    log(f"{path}: beta {np.round(beta, 3).tolist()}, removal shifts "
        f"{[round(v, 4) for v in res['vote_removal_shift'].values()]}")
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    A_fit, A_ev = patterns(m, fit_w), patterns(m, ev_w)
    lp_ref = logprobs(m, ev_w)
    rule_fit, rule_ev = Rule(fit_w, shapes), Rule(ev_w, shapes)
    w = fv_weights(m)
    tt = C.tt_tables(w)
    # M5: distance profile
    res["distance_profile"] = {h: [float(A_fit[:, h, np.arange(C.D_MAX, C.T), np.arange(C.D_MAX, C.T) - d].mean())
                                   for d in range(1, C.D_MAX + 1)] for h in range(H)}
    res["peak_distance"] = {h: 1 + int(np.argmax(res["distance_profile"][h])) for h in range(H)}
    res["heads"] = {}
    for h in range(H):
        t0 = time.time()
        own = names[h]
        arms = {"base": ({}, {}), "own": (u_features(fit_w, own, beta[h]), u_features(ev_w, own, beta[h]))}
        for t in TABLE_NAMES:
            if t == own:
                continue
            key = f"other_{t}" if t in names else f"alt_{t}"
            if key.startswith("alt_") and not (own in ("skip2", "skip3") and t in ("skip2", "skip3")):
                continue
            arms[key] = (u_features(fit_w, t, beta[h]), u_features(ev_w, t, beta[h]))
        arms["perm"] = (u_features(fit_w, own, beta[h], "Up"), u_features(ev_w, own, beta[h], "Up"))
        arms["tt"] = ({"c": torch.as_tensor(tt_feature(fit_w, tt[h]))}, {"c": torch.as_tensor(tt_feature(ev_w, tt[h]))})
        arms["tt_self"] = ({"c": torch.as_tensor(tt_self_feature(fit_w, tt[h]))},
                           {"c": torch.as_tensor(tt_self_feature(ev_w, tt[h]))})
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        A_e = torch.as_tensor(A_ev[:, h].astype(np.float64))
        hr = {"table": own, "pattern_entropy_fit": entropy(A_f), "arms": {}}
        for name, (ff, fe) in arms.items():
            theta, loss, g = fit_rule(A_f, rule_fit, ff)
            with torch.no_grad():
                pat = torch.exp(rule_ev.logpattern(theta, fe)).numpy()
            lp = logprobs(m, ev_w, h, pat)
            hr["arms"][name] = {"kl": C.kl(lp_ref, lp), "kl_win": [float(x) for x in C.kl_per_window(lp_ref, lp)],
                                "pattern_ce_fit": loss, "pattern_ce_eval": pattern_ce(A_e, rule_ev, theta, fe),
                                "max_grad": g,
                                "coef": {k: float(v) for k, v in theta.items() if k not in ("beta", "sk")}}
        base = hr["arms"]["base"]["kl"]
        for name in hr["arms"]:
            hr["arms"][name]["gain"] = base - hr["arms"][name]["kl"]
        res["heads"][h] = hr
        log(f"  head {h} ({own}): base KL {base:.5f}; gains " +
            ", ".join(f"{k} {v['gain']:+.5f}" for k, v in hr["arms"].items() if k != "base") + f" [{time.time() - t0:.0f}s]")
        del arms
    # M4: pure-pair correlation at d = 1
    Y = C.pure_pair_scores(w)
    N1 = np.load(os.path.join(C.REPO, "analysis", "out", "candidates", "qk", "N_1.npy")).astype(np.float64)
    kidx, qidx = np.nonzero(N1 >= C.MIN_COUNT)
    wts = N1[kidx, qidx]
    cache = {}

    def resid(key, table):
        if key not in cache:
            cache[key] = double_centre(table, kidx, qidx, wts)[0]
        return cache[key]

    for h in range(H):
        ry = resid(f"score_h{h}", Y[h])
        pc = {}
        for t in TABLE_NAMES:
            pc[t] = wcorr(ry, resid(t, usefulness_tables(t)[0][1]), wts)
        pc["perm_own"] = wcorr(ry, resid(f"perm_{names[h]}", np.load(os.path.join(U_DIR, f"Up_{names[h]}_d1.npy"))),
                               wts)
        res["heads"][h]["pure_pair_corr"] = pc
        log(f"  head {h} ({names[h]}): pure-pair corr " + ", ".join(f"{k} {v:+.3f}" for k, v in pc.items()))
    with open(os.path.join(out_dir, "law.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
