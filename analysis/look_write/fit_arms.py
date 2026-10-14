"""Per model: LOCAL/RECALL classification, base attention rule and pair-feature arms (fitted to each head's own
pattern on train_windows(32, 2024)), exact-model substitution output test on train_windows(32, 2025), and the
double-centred pure-pair correlations at d = 1.

Base rule (per head), score for query i, key j <= i:
    s(i, j) = beta[bucket(i - j)] [j < i]                  9 distance buckets 1,2,3,4,5-7,8-15,16-31,32-63,64+
            + sigma[shape(t_i)] [j = i]                     self x query shape (4)
            + kappa[shape(t_j)] [j < i]                     key shape (4)
The distance-0 bucket of the pre-registration is dropped because it equals the sum of the four self x shape
indicators (identical function class). Arms add
    c1 f_1(t_j, t_i) [i - j = 1] + c2 f_{i-j}(t_j, t_i) [2 <= i - j <= 7]          (UOV-type features)
or  c TT(t_i, t_j) [j < i]                                                          (exact token-token ceiling).
Loss: mean over queries i >= 1 of the pattern cross-entropy -sum_j A_ij log softmax_j s(i, j) (convex), L-BFGS.

Run: OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write.fit_arms attn_s1 ...
     exploratory: ... --band 8 63   (needs uov_tables.py --band 8 63 first; writes arms_uovband.json)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.look_write import common as C


# ----------------------------------------------------------------------------- classification

def mean_distances(A):
    """A: (n, H, T, T). Mean attention distance to earlier tokens (j < i), weighted by attention, queries i >= 1.
    pooled: sum A_ij (i-j) / sum A_ij over all (i, j < i);  per_query: per-query renormalised over j < i, then the
    mean over queries whose earlier-token mass is > 0."""
    T = A.shape[-1]
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    lower = (dist > 0).astype(np.float64)
    out = []
    for h in range(A.shape[1]):
        a = A[:, h].astype(np.float64) * lower
        mass = a.sum(-1)[:, 1:]                     # (n, T-1)
        num = (a * dist).sum(-1)[:, 1:]
        pooled = num.sum() / mass.sum()
        ok = mass > 1e-12
        per_q = (num[ok] / mass[ok]).mean()
        out.append({"pooled": float(pooled), "per_query": float(per_q), "earlier_mass": float(mass.mean())})
    return out


# ----------------------------------------------------------------------------- features

def bucket_index(T):
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    idx = np.full((T, T), len(C.BUCKET_EDGES), dtype=np.int64)          # dummy index for j >= i (weight fixed 0)
    for b, (lo, hi) in enumerate(C.BUCKET_EDGES):
        idx[(dist >= lo) & (dist <= hi)] = b
    return idx


def shape_index(wins, shapes):
    """c2[b, i, j] = shape(t_i) for j = i, 4 + shape(t_j) for j < i, 8 (dummy) for j > i."""
    n, T = wins.shape
    sh = shapes[wins]
    c2 = np.full((n, T, T), 8, dtype=np.int64)
    low = np.tril(np.ones((T, T), dtype=bool), -1)
    c2[:, low] = (4 + np.broadcast_to(sh[:, None, :], (n, T, T)))[:, low]
    ii = np.arange(T)
    c2[:, ii, ii] = sh
    return c2


def pair_features(wins, tables):
    """tables: dict d -> (V, V) [k, q]. Returns (F1, F2): F1[b, i, i-1] = f_1(t_{i-1}, t_i);
    F2[b, i, i-d] = f_d(t_{i-d}, t_i) for d = 2..7. float64 (n, T, T)."""
    n, T = wins.shape
    F1 = np.zeros((n, T, T))
    F2 = np.zeros((n, T, T))
    for d in range(1, C.D_MAX + 1):
        i = np.arange(d, T)
        vals = tables[d][wins[:, i - d], wins[:, i]]
        (F1 if d == 1 else F2)[:, i, i - d] = vals
    return F1, F2


def band_feature(wins, table, lo, hi):
    """Exploratory: Fb[b, i, j] = f(t_j, t_i) for lo <= i - j <= hi (table [k, q]), else 0."""
    n, T = wins.shape
    F = table[wins[:, None, :], wins[:, :, None]].astype(np.float64)
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    F *= ((dist >= lo) & (dist <= hi))[None]
    return F


def tt_feature(wins, tt):
    """FTT[b, i, j] = TT(t_i, t_j) for j < i (tt indexed [q, k])."""
    n, T = wins.shape
    F = tt[wins[:, :, None], wins[:, None, :]].astype(np.float64)
    F *= np.tril(np.ones((T, T)), -1)[None]
    return F


# ----------------------------------------------------------------------------- the log-linear rule

class Rule:
    """Score tensor builder for one set of windows."""

    def __init__(self, wins, shapes):
        n, T = wins.shape
        self.n, self.T = n, T
        self.c1 = torch.as_tensor(bucket_index(T))
        self.c2 = torch.as_tensor(shape_index(wins, shapes))
        self.mask = torch.ones(T, T, dtype=torch.bool).tril()

    def scores(self, theta, feats):
        beta = torch.cat([theta["beta"], theta["beta"].new_zeros(1)])
        sk = torch.cat([theta["sk"], theta["sk"].new_zeros(1)])
        s = beta[self.c1][None] + sk[self.c2]
        for name, F in feats.items():
            s = s + theta[name] * F
        return s.masked_fill(~self.mask, -1e30)

    def logpattern(self, theta, feats):
        return torch.log_softmax(self.scores(theta, feats), -1)


def fit_rule(A, rule, feats, max_iter=2000):
    """A: (n, T, T) float64 tensor (target pattern). feats: dict name -> (n, T, T) tensor. Returns theta, loss."""
    theta = {"beta": torch.zeros(len(C.BUCKET_EDGES), dtype=torch.float64, requires_grad=True),
             "sk": torch.zeros(8, dtype=torch.float64, requires_grad=True)}
    for name in feats:
        theta[name] = torch.zeros((), dtype=torch.float64, requires_grad=True)
    nq = A.shape[0] * (A.shape[1] - 1)
    opt = torch.optim.LBFGS(list(theta.values()), lr=1.0, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-13, history_size=50, line_search_fn="strong_wolfe")
    low = rule.mask[None]

    def closure():
        opt.zero_grad()
        lp = rule.logpattern(theta, feats)
        loss = -(torch.where(low, A * lp, torch.zeros_like(lp))).sum() / nq
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        lp = rule.logpattern(theta, feats)
        loss = float(-(torch.where(low, A * lp, torch.zeros_like(lp))).sum() / nq)
        g = max(float(p.grad.abs().max()) for p in theta.values())
    return {k: v.detach().clone() for k, v in theta.items()}, loss, g


def pattern_ce(A, rule, theta, feats):
    nq = A.shape[0] * (A.shape[1] - 1)
    with torch.no_grad():
        lp = rule.logpattern(theta, feats)
        return float(-(torch.where(rule.mask[None], A * lp, torch.zeros_like(lp))).sum() / nq)


def entropy(A):
    a = A.double()
    return float(-(torch.where(a > 0, a * torch.log(a.clamp_min(1e-300)), torch.zeros_like(a))).sum()
                 / (A.shape[0] * (A.shape[1] - 1)))


# ----------------------------------------------------------------------------- pure-pair correlation

def double_centre(Y, kidx, qidx, w, iters=3000, tol=1e-13):
    """Weighted least-squares additive fit y ~ a_k + b_q over the listed cells; returns the residual vector.
    Preconditioned conjugate gradient on the normal equations (singular but consistent; residual is unique)."""
    V = C.V
    y = Y[kidx, qidx].astype(np.float64)
    wk = np.bincount(kidx, w, V)
    wq = np.bincount(qidx, w, V)
    diag = np.concatenate([wk, wq])
    pinv = np.where(diag > 0, 1.0 / np.maximum(diag, 1e-300), 0.0)

    def Amul(t):
        r = w * (t[kidx] + t[V + qidx])
        return np.concatenate([np.bincount(kidx, r, V), np.bincount(qidx, r, V)])

    rhs = np.concatenate([np.bincount(kidx, w * y, V), np.bincount(qidx, w * y, V)])
    x = np.zeros(2 * V)
    r = rhs.copy()
    z = pinv * r
    p = z.copy()
    rz = r @ z
    r0 = np.sqrt(rhs @ rhs)
    for it in range(iters):
        Ap = Amul(p)
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.sqrt(r @ r) <= tol * r0:
            break
        z = pinv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    res = y - x[kidx] - x[V + qidx]
    return res, it, float(np.sqrt(r @ r) / r0)


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


# ----------------------------------------------------------------------------- per model

def load_tables(model_id, prefix, h):
    d = C.model_dir(model_id)
    return {dd: np.load(os.path.join(d, f"{prefix}_h{h}_d{dd}.npy")) for dd in range(1, C.D_MAX + 1)}


def run_model(model_id, uov_prefix="uov", band=None, log=print):
    C.set_threads()
    out_dir = C.model_dir(model_id)
    m = C.load(model_id)
    w = C.weights(m)
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    A_fit = C.patterns(m, fit_w)
    A_ev = C.patterns(m, ev_w)
    dists = mean_distances(A_fit)
    kinds = ["LOCAL" if dd["pooled"] <= C.LOCAL_MAX_DIST else "RECALL" for dd in dists]
    log(f"{model_id}: mean distances {[round(x['pooled'], 2) for x in dists]} -> {kinds}")
    lp_ref = C.logprobs_sub(m, ev_w)
    rule_fit, rule_ev = Rule(fit_w, shapes), Rule(ev_w, shapes)
    tt = C.tt_tables(w)

    # feature tensors for every source (own/other UOV, permuted UOV, TT), on fit and eval windows
    def uov_feats(prefix, h):
        if band is not None:      # exploratory: one band table per head, one coefficient
            tab = np.load(os.path.join(out_dir, f"{prefix}band_h{h}.npy"))
            return ({"cb": torch.as_tensor(band_feature(fit_w, tab, *band))},
                    {"cb": torch.as_tensor(band_feature(ev_w, tab, *band))})
        tabs = load_tables(model_id, prefix, h)
        f = pair_features(fit_w, tabs)
        e = pair_features(ev_w, tabs)
        return ({"c1": torch.as_tensor(f[0]), "c2": torch.as_tensor(f[1])},
                {"c1": torch.as_tensor(e[0]), "c2": torch.as_tensor(e[1])})

    uov = {h: uov_feats(uov_prefix, h) for h in range(C.H)}
    res = {"model": model_id, "final_ln_scale": meta["final_ln_scale"], "uov_prefix": uov_prefix, "band": band,
           "heads": {}}
    for h in range(C.H):
        t0 = time.time()
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        A_e = torch.as_tensor(A_ev[:, h].astype(np.float64))
        arms = {"base": ({}, {})}
        arms["own"] = uov[h]
        for h2 in range(C.H):
            if h2 != h:
                arms[f"other_h{h2}"] = uov[h2]
        arms["perm"] = uov_feats(uov_prefix.replace("uov", "uovp", 1), h)
        arms["tt"] = ({"c": torch.as_tensor(tt_feature(fit_w, tt[h]))}, {"c": torch.as_tensor(tt_feature(ev_w, tt[h]))})
        hr = {"kind": kinds[h], "mean_dist": dists[h], "pattern_entropy_fit": entropy(A_f),
              "pattern_entropy_eval": entropy(A_e), "arms": {}}
        for name, (ff, fe) in arms.items():
            theta, loss, g = fit_rule(A_f, rule_fit, ff)
            with torch.no_grad():
                pat = torch.exp(rule_ev.logpattern(theta, fe)).float().numpy()
            lp = C.logprobs_sub(m, ev_w, h, pat)
            hr["arms"][name] = {
                "kl": C.kl(lp_ref, lp), "kl_win": [float(x) for x in C.kl_per_window(lp_ref, lp)],
                "pattern_ce_fit": loss, "pattern_ce_eval": pattern_ce(A_e, rule_ev, theta, fe),
                "max_grad": g, "coef": {k: float(v) for k, v in theta.items() if k not in ("beta", "sk")},
                "beta": [float(x) for x in theta["beta"]], "shape_terms": [float(x) for x in theta["sk"]]}
        base = hr["arms"]["base"]["kl"]
        for name in hr["arms"]:
            hr["arms"][name]["gain"] = base - hr["arms"][name]["kl"]
        res["heads"][h] = hr
        log(f"  head {h} ({kinds[h]}): base KL {base:.5f}; gains " +
            ", ".join(f"{k} {v['gain']:+.5f}" for k, v in hr["arms"].items() if k != "base") +
            f"; own coef {hr['arms']['own']['coef']} [{time.time() - t0:.0f}s]")

    if band is not None:
        with open(os.path.join(out_dir, "arms_uovband.json"), "w") as f:
            json.dump(res, f, indent=1)
        return res
    # pure-pair correlations at d = 1
    Y = C.pure_pair_scores(w)
    N1 = np.load(os.path.join(out_dir, "N_1_recount.npy")).astype(np.float64)
    kidx, qidx = np.nonzero(N1 >= C.MIN_COUNT)
    wts = N1[kidx, qidx]
    res["pure_pair"] = {"n_pairs": int(len(kidx)), "total_weight": float(wts.sum())}
    d1 = {h: np.load(os.path.join(out_dir, f"{uov_prefix}_h{h}_d1.npy")) for h in range(C.H)}
    d1p = {h: np.load(os.path.join(out_dir, f"{uov_prefix.replace('uov', 'uovp', 1)}_h{h}_d1.npy"))
           for h in range(C.H)}
    resid_cache = {}

    def resid(name, table):
        if name not in resid_cache:
            r, it, rel = double_centre(table, kidx, qidx, wts)
            resid_cache[name] = r
            res["pure_pair"].setdefault("cg", {})[name] = [it, rel]
        return resid_cache[name]

    for h in range(C.H):
        ry = resid(f"score_h{h}", Y[h])
        pc = {"own": wcorr(ry, resid(f"uov_h{h}", d1[h]), wts),
              "perm": wcorr(ry, resid(f"uovp_h{h}", d1p[h]), wts),
              "tt": wcorr(ry, resid(f"tt_h{h}", tt[h].T), wts)}
        for h2 in range(C.H):
            if h2 != h:
                pc[f"other_h{h2}"] = wcorr(ry, resid(f"uov_h{h2}", d1[h2]), wts)
        res["heads"][h]["pure_pair_corr"] = pc
        log(f"  head {h}: pure-pair corr " + ", ".join(f"{k} {v:+.3f}" for k, v in pc.items()))
    tag = "" if uov_prefix == "uov" else "_" + uov_prefix
    with open(os.path.join(out_dir, f"arms{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--uov-prefix", default="uov")
    ap.add_argument("--band", nargs=2, type=int, default=None, help="exploratory distance-band UOV (e.g. 8 63)")
    a = ap.parse_args()
    for mid in a.models:
        t0 = time.time()
        run_model(mid, a.uov_prefix, a.band)
        print(f"{mid}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
