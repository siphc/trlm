"""Usefulness under transformed skip-gram votes (prefix qk), d = 1..7, vote_k(v) = f(PMI_{d+1}(k, v)):
  UPOS  f = max(PMI, 0)                  (positive PMI: votes only for words k makes likelier)
  UCLIP f = max(PMI, -1)                 (PMI with a floor at -1)
  UPROB f = P_{d+1}(v | k) = exp(PMI) P(v) (a probability vote)
U_f = (sum f(k, next) - N_d E_{P1(.|q)} f(k, .)) / (N_d + 20)."""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "qk")
V, CHUNK, DM = 2000, 20_000_000, 7
FS = {"UPOS": lambda p, pu: np.maximum(p, 0.0), "UCLIP": lambda p, pu: np.maximum(p, -1.0),
      "UPROB": lambda p, pu: np.exp(p) * pu[None, :]}


def main():
    from analysis.mech_eval import stats
    from harness.common import load_corpus

    t0 = time.time()
    s = stats()
    pu = np.asarray(s.p_uni, np.float64)
    P1 = np.exp(np.asarray(s.log_follow, np.float64))
    corpus = load_corpus()
    n = len(corpus)
    for d in range(1, DM + 1):
        pm = np.asarray(s.pmi[d + 1], np.float64)
        tabs = {nm: f(pm, pu) for nm, f in FS.items()}
        flat = {nm: t.astype(np.float32).reshape(-1) for nm, t in tabs.items()}
        S = {nm: np.zeros(V * V) for nm in FS}
        for start in range(0, n - DM - 1, CHUNK):
            length = min(CHUNK, n - DM - 1 - start)
            x = np.asarray(corpus[start:start + length + DM + 1]).astype(np.int64)
            k = x[:length]
            pair = k * V + x[d:d + length]
            kv = k * V + x[d + 1:d + 1 + length]
            for nm in FS:
                S[nm] += np.bincount(pair, weights=flat[nm][kv], minlength=V * V)
        N = np.load(os.path.join(OUT, f"N_{d}.npy")).astype(np.float64)
        for nm in FS:
            B = (P1 @ tabs[nm].T).T
            np.save(os.path.join(OUT, f"{nm}_{d}.npy"), ((S[nm].reshape(V, V) - N * B) / (N + 20)).astype(np.float32))
        print(d, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
