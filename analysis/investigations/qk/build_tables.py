"""Pair tables [k, q] (k sits d tokens before q) for attention features, from the counts of pairstats.py.

Corpus (closed-form counting rules, TRAIN stream):
  N_d, S_d          occurrences of (k, q) at distance d; sum of PMI_{d+1}(k, next token after q) over them
  B_d[k, q]         = sum_v P1(v | q) PMI_{d+1}(k, v)          (k's vote scored by what q alone predicts)
  U_d = (S_d - N_d B_d) / (N_d + KAPPA)                         vote usefulness (shrunk to 0 for rare pairs)
  A_d = (S_d + KAPPA B_d) / (N_d + KAPPA)                       vote fit (shrunk towards B)
  G_d = A_d - log sum_v P1(v | q) exp(PMI_{d+1}(k, v))          expected log-likelihood gain of k's vote
  BACK_d = log((N_d + 10 P(k)) / (sum_k' N_d[k', q] + 10))      backward: how predictable k is given q
  IG_1  = KL(P3(. | k, q) || P1(. | q))                         trigram information gain (d = 1)
Weights (insight only; never in a candidate): UOV_h_d = (SOV_h_d - N_d BOV_h) / (N_d + KAPPA), h = 0, 2, d = 1..3.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "qk")
V = 2000
KAPPA = 20.0


def main():
    from analysis.mech_eval import stats
    sys.path.insert(0, os.path.dirname(__file__))
    from qk_pairs import trigram_ig

    t0 = time.time()
    s = stats()
    P1 = np.exp(np.asarray(s.log_follow, np.float64))
    pu = np.asarray(s.p_uni, np.float64)
    for d in range(1, 8):
        N = np.load(os.path.join(OUT, f"N_{d}.npy")).astype(np.float64)
        S = np.load(os.path.join(OUT, f"S_{d}.npy")).astype(np.float64)
        pv = np.asarray(s.pmi[d + 1], np.float64)
        B = (P1 @ pv.T).T
        LSE = np.log(P1 @ np.exp(pv).T).T
        A = (S + KAPPA * B) / (N + KAPPA)
        np.save(os.path.join(OUT, f"U_{d}.npy"), ((S - N * B) / (N + KAPPA)).astype(np.float32))
        np.save(os.path.join(OUT, f"A_{d}.npy"), A.astype(np.float32))
        np.save(os.path.join(OUT, f"G_{d}.npy"), (A - LSE).astype(np.float32))
        np.save(os.path.join(OUT, f"BACK_{d}.npy"),
                np.log((N + 10 * pu[:, None]) / (N.sum(0, keepdims=True) + 10)).astype(np.float32))
        if d <= 3:
            for h in (0, 2):
                OV = np.load(os.path.join(OUT, f"ov_h{h}.npy")).astype(np.float64)
                SO = np.load(os.path.join(OUT, f"SOV_h{h}_{d}.npy")).astype(np.float64)
                BO = (P1 @ OV.T).T
                np.save(os.path.join(OUT, f"UOV_h{h}_{d}.npy"), ((SO - N * BO) / (N + KAPPA)).astype(np.float32))
        print(d, time.time() - t0, flush=True)
    np.save(os.path.join(OUT, "IG_1.npy"), trigram_ig(s))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
