"""Usefulness tables for the program's own votes (Track B of the look-follows-write test, 2026-10-14).

For a vote table X (X[k, v]: what an earlier token k votes for v) and distance d (k sits d tokens before the current
token q), the usefulness of k's vote given q is

    U^X_d(k, q) = (S^X_d[k, q] - N_d[k, q] * B^X[k, q]) / (N_d[k, q] + 20)
    S^X_d[k, q] = sum over corpus positions j with x_j = k, x_{j+d} = q of X[k, x_{j+d+1}]   (what actually came next)
    B^X[k, q]   = sum_v P1(v | q) X[k, v]                                                    (what q alone predicts)

U is linear in X, so the usefulness of a head's whole vote is the strength-weighted sum of the tables' U. Tables:
  skip    PMI_{d+1} (S and N reused from round 2: analysis/out/candidates/qk/S_d.npy, N_d.npy)
  succ    PMI_1     (earlier token's own successors; LOCAL-0)
  alike   slot affinity (LOCAL-0)
  copy    one-hot: X[k, v] = [v = k] (LOCAL-2's repeat veto); S = #(x_j = k, x_{j+d} = q, x_{j+d+1} = k)
Closed-form counts of the TRAIN stream plus the program's P1 and PMI tables; no model numbers.
Output: analysis/out/candidates/useful/U_{table}_{d}.npy (float32, (V, V)), d = 1..7.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
QK = os.path.join(REPO, "analysis", "out", "candidates", "qk")
OUT = os.path.join(REPO, "analysis", "out", "candidates", "useful")
V, CHUNK, DM, PRIOR = 2000, 20_000_000, 7, 20.0


def main():
    from analysis.mech_eval import stats
    from harness.common import load_corpus

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    s = stats()
    X = {"succ": np.asarray(s.pmi[1], np.float64).reshape(-1), "alike": np.asarray(s.slot_affinity, np.float64).reshape(-1)}
    S = {(t, d): np.zeros(V * V) for t in ("succ", "alike", "copy") for d in range(1, DM + 1)}
    corpus = load_corpus()
    n = len(corpus)
    for start in range(0, n - DM - 1, CHUNK):
        length = min(CHUNK, n - DM - 1 - start)
        x = np.asarray(corpus[start:start + length + DM + 1]).astype(np.int64)
        k = x[:length]
        for d in range(1, DM + 1):
            pair = k * V + x[d:d + length]
            nxt = x[d + 1:d + 1 + length]
            kv = k * V + nxt
            for t in ("succ", "alike"):
                S[(t, d)] += np.bincount(pair, weights=X[t][kv], minlength=V * V)
            S[("copy", d)] += np.bincount(pair[nxt == k], minlength=V * V)
        print(f"{start + length:,} tokens {time.time() - t0:.0f}s", flush=True)
    P1 = np.exp(np.asarray(s.log_follow, np.float64))                   # P1[q, v]
    B = {"succ": X["succ"].reshape(V, V) @ P1.T, "alike": X["alike"].reshape(V, V) @ P1.T, "copy": P1.T.copy()}
    B["skip"] = None
    for d in range(1, DM + 1):
        N = np.load(os.path.join(QK, f"N_{d}.npy")).astype(np.float64)
        for t in ("succ", "alike", "copy"):
            U = (S[(t, d)].reshape(V, V) - N * B[t]) / (N + PRIOR)
            np.save(os.path.join(OUT, f"U_{t}_{d}.npy"), U.astype(np.float32))
        pmi = np.asarray(s.pmi[min(d + 1, 8)], np.float64)
        Sk = np.load(os.path.join(QK, f"S_{d}.npy")).astype(np.float64)
        U = (Sk - N * (pmi @ P1.T)) / (N + PRIOR)
        np.save(os.path.join(OUT, f"U_skip_{d}.npy"), U.astype(np.float32))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
