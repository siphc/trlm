"""Reverse question (insight): which corpus statistic best predicts the exact query-key pair interaction of heads 0
and 2 for a key d tokens back (d = 1, 2)?

Exact score of key token k at distance d for query token q (head h), averaged over query positions POS:
    S_d(q, k) = LN1(e_q + p_i) W_Q . LN1(e_k + p_{i-d}) W_K / sqrt(32)
Target: R_d(q, k) = S_d(q, k) - S_0(q) (relative to the current token itself, which the softmax compares against).
Pairs: (k, q) with k exactly d tokens before q in the TRAIN stream, count >= MIN_COUNT; weights = counts.
Three views: raw; row-centred (query effects removed); double-centred (query and key effects removed by a weighted
additive fit), i.e. the pure pair interaction.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "qk")
V, D, DH = 2000, 128, 32
POS = [16, 48, 96, 160, 224]
MIN_COUNT = 20
KAPPA = 20.0


def exact_scores(h, d):
    from analysis.tools import Model
    m = Model("attn_s0")
    WQ = m.Wqkv[h * DH:(h + 1) * DH]
    WK = m.Wqkv[D + h * DH:D + (h + 1) * DH]
    S = np.zeros((V, V))
    S0 = np.zeros(V)
    for i in POS:
        hq, _ = m.ln(m.WE + m.WP[i], m.g1)
        hk, _ = m.ln(m.WE + m.WP[i - d], m.g1)
        Q, K = hq @ WQ.T, hk @ WK.T
        S += Q @ K.T / np.sqrt(DH)
        S0 += (Q * (hq @ WK.T)).sum(1) / np.sqrt(DH)
    return S / len(POS), S0 / len(POS)         # S[q, k]


def trigram_ig(s):
    """IG(k, q) = KL(P3(. | k, q) || P1(. | q)), P3 Dirichlet-smoothed (KAPPA) towards P1. Returns dict pair->IG."""
    keys = np.asarray(s.tri_keys)
    cnt = np.asarray(s.tri_counts).astype(np.float64)
    pair = keys // V
    nxt = keys % V
    q = pair % V
    starts = np.r_[0, np.nonzero(np.diff(pair))[0] + 1]
    Np = np.add.reduceat(cnt, starts)
    rep = np.repeat(Np, np.diff(np.r_[starts, len(keys)]))
    P1 = np.exp(np.asarray(s.log_follow, np.float64))
    p1 = P1[q, nxt]
    ph = (cnt + KAPPA * p1) / (rep + KAPPA)
    seen = np.add.reduceat(ph * (np.log(ph) - np.log(p1)), starts)
    p1seen = np.add.reduceat(p1, starts)
    fac = KAPPA / (Np + KAPPA)
    ig = seen + fac * np.clip(1 - p1seen, 0, None) * np.log(fac)
    out = np.zeros(V * V, np.float32)
    out[pair[starts]] = ig
    return out.reshape(V, V)      # [k, q]


def wcorr(u, v, w):
    u = u - np.average(u, weights=w)
    v = v - np.average(v, weights=w)
    return float(np.average(u * v, weights=w) / np.sqrt(np.average(u * u, weights=w) * np.average(v * v, weights=w) + 1e-30))


def center(y, w, q, k, double, iters=30):
    y = y.copy()
    for _ in range(iters if double else 1):
        rm = np.bincount(q, weights=w * y, minlength=V) / np.maximum(np.bincount(q, weights=w, minlength=V), 1e-12)
        y -= rm[q]
        if not double:
            break
        cm = np.bincount(k, weights=w * y, minlength=V) / np.maximum(np.bincount(k, weights=w, minlength=V), 1e-12)
        y -= cm[k]
    return y


def main():
    from analysis.mech_eval import stats
    from program.vocab import TOKENS

    t0 = time.time()
    s = stats()
    P1 = np.exp(np.asarray(s.log_follow, np.float64))              # [q, v]
    big = np.asarray(s.bigram_counts)
    res = {}
    ig = trigram_ig(s)
    print("IG done", time.time() - t0, flush=True)
    for d in (1, 2):
        N = np.load(os.path.join(OUT, f"N_{d}.npy")).astype(np.float64)   # [k, q]
        Ssum = np.load(os.path.join(OUT, f"S_{d}.npy")).astype(np.float64)
        pmi_vote = np.asarray(s.pmi[min(d + 1, 8)], np.float64)            # [k, v]
        B = (P1 @ pmi_vote.T).T                                            # [k, q] = sum_v P1(v|q) PMI(k, v)
        LSE = np.log(P1 @ np.exp(pmi_vote).T).T                            # [k, q]
        A = Ssum / np.maximum(N, 1)
        U = (Ssum - N * B) / (N + KAPPA)
        G = (Ssum + KAPPA * B) / (N + KAPPA) - LSE
        kk, qq = np.nonzero(N >= MIN_COUNT)
        w = N[kk, qq]
        feats = {
            "PMI_d(k,q)": np.asarray(s.pmi[d], np.float64)[kk, qq],
            "log P(q|k) at d": np.log(N[kk, qq] / N.sum(1)[kk]),
            "log P(k|q) backward": np.log(N[kk, qq] / N.sum(0)[qq]),
            "vote fit A=E[PMI(k,next)|k,q]": A[kk, qq],
            "vote vs bigram B=E_P1(.|q)[PMI(k,.)]": B[kk, qq],
            "vote usefulness U=A-B (shrunk)": U[kk, qq],
            "vote log-lik gain G": G[kk, qq],
            "key info_{d+1}(k)": np.asarray(s.info)[min(d, 7), kk],
            "cur is piece x key cont": np.asarray(s.piece)[qq] * np.asarray(s.cont)[kk],
        }
        if d == 1:
            feats["trigram IG KL(P3||P1)"] = ig[kk, qq].astype(np.float64)
        for h in (0, 2):
            ovp = os.path.join(OUT, f"SOV_h{h}_{d}.npy")
            if os.path.exists(ovp):
                OV = np.load(os.path.join(OUT, f"ov_h{h}.npy")).astype(np.float64)
                SO = np.load(ovp).astype(np.float64)
                BO = (P1 @ OV.T).T
                feats[f"[weights] OV_h{h} usefulness"] = ((SO - N * BO) / (N + KAPPA))[kk, qq]
                feats[f"[weights] OV_h{h} fit A"] = (SO / np.maximum(N, 1))[kk, qq]
        for h in (0, 2):
            Sx, S0 = exact_scores(h, d)
            y = Sx[qq, kk] - S0[qq]
            r = {"n_pairs": int(len(kk)), "weight": float(w.sum())}
            views = {}
            for view, dbl in (("raw", None), ("row", False), ("double", True)):
                yv = y if dbl is None else center(y, w, qq, kk, dbl)
                views[view] = {"target_sd": float(np.sqrt(np.average((yv - np.average(yv, weights=w)) ** 2, weights=w)))}
                for name, f in feats.items():
                    fv = f if dbl is None else center(f, w, qq, kk, dbl)
                    views[view][name] = round(wcorr(yv, fv, w), 3)
            # share of variance: additive (query + key) vs pair
            yr = center(y, w, qq, kk, False)
            yd = center(y, w, qq, kk, True)
            var = lambda z: float(np.average((z - np.average(z, weights=w)) ** 2, weights=w))
            r["var_raw"], r["var_after_query"], r["var_pure_pair"] = var(y), var(yr), var(yd)
            r["views"] = views
            # biggest pure-pair interactions by attention-relevant mass
            order = np.argsort(-w * np.abs(yd))[:25]
            r["top_pairs"] = [(TOKENS[kk[o]], TOKENS[qq[o]], round(float(yd[o]), 2), int(w[o]),
                               round(float(feats["vote usefulness U=A-B (shrunk)"][o]), 2),
                               round(float(feats["PMI_d(k,q)"][o]), 2)) for o in order]
            res[f"h{h}_d{d}"] = r
            print(f"h{h} d={d}: {time.time() - t0:.0f}s", flush=True)
            print(json.dumps({v: views[v] for v in views}, indent=1), flush=True)
    with open(os.path.join(REPO, "analysis", "investigations", "qk", "qk_pairs.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
