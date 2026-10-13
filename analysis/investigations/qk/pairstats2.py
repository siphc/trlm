"""Usefulness under alternative corpus votes (prefix qk), d = 1..7, pairs (k d tokens before q):
  UBIG_d  = vote PMI_1(k, .) (k's bigram, at every distance):  (sum PMI_1(k, next) - N B1) / (N + 20)
  UCOPY_d = vote 'not k again' (a veto on repeating k):       (N P1(k|q) - #[next == k]) / (N + 20)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "qk")
V, CHUNK, DM = 2000, 20_000_000, 7


def main():
    from analysis.mech_eval import stats
    from harness.common import load_corpus

    t0 = time.time()
    s = stats()
    pmi1 = np.ascontiguousarray(np.asarray(s.pmi[1], np.float32)).reshape(-1)
    corpus = load_corpus()
    n = len(corpus)
    SB = {d: np.zeros(V * V) for d in range(1, DM + 1)}
    C = {d: np.zeros(V * V) for d in range(1, DM + 1)}
    for start in range(0, n - DM - 1, CHUNK):
        length = min(CHUNK, n - DM - 1 - start)
        x = np.asarray(corpus[start:start + length + DM + 1]).astype(np.int64)
        k = x[:length]
        for d in range(1, DM + 1):
            pair = k * V + x[d:d + length]
            nxt = x[d + 1:d + 1 + length]
            SB[d] += np.bincount(pair, weights=pmi1[k * V + nxt], minlength=V * V)
            C[d] += np.bincount(pair[nxt == k], minlength=V * V)
        print(f"{start + length:,} {time.time() - t0:.0f}s", flush=True)
    P1 = np.exp(np.asarray(s.log_follow, np.float64))
    pm1 = np.asarray(s.pmi[1], np.float64)
    B1 = (P1 @ pm1.T).T                       # [k, q]
    Pk = P1.T                                 # [k, q] = P1(k | q)
    for d in range(1, DM + 1):
        N = np.load(os.path.join(OUT, f"N_{d}.npy")).astype(np.float64)
        np.save(os.path.join(OUT, f"UBIG_{d}.npy"), ((SB[d].reshape(V, V) - N * B1) / (N + 20)).astype(np.float32))
        np.save(os.path.join(OUT, f"UCOPY_{d}.npy"), ((N * Pk - C[d].reshape(V, V)) / (N + 20)).astype(np.float32))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
