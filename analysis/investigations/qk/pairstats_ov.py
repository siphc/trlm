"""Insight only (weights): S^OV_h,d[k, q] = sum over corpus occurrences (k d tokens before q) of OV_h[k, next],
for heads 0, 2, 3 and d = 1..7; then UOV_h_d = (S - N_d B^OV_h) / (N_d + 20) with B^OV_h[k, q] = sum_v P1(v|q) OV_h[k, v]."""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairstats import OUT, V, ov_table  # noqa: E402

CHUNK = 20_000_000
HEADS = (0, 2, 3)
DM = 7


def main():
    from analysis.mech_eval import stats
    from harness.common import load_corpus

    t0 = time.time()
    ov = {}
    for h in HEADS:
        ov[h] = ov_table(h)
        np.save(os.path.join(OUT, f"ov_h{h}.npy"), ov[h])
        ov[h] = ov[h].reshape(-1)
    corpus = load_corpus()
    n = len(corpus)
    SO = {(h, d): np.zeros(V * V, np.float64) for h in HEADS for d in range(1, DM + 1)}
    for start in range(0, n - DM - 1, CHUNK):
        length = min(CHUNK, n - DM - 1 - start)
        x = np.asarray(corpus[start:start + length + DM + 1]).astype(np.int64)
        k = x[:length]
        for d in range(1, DM + 1):
            pair = k * V + x[d:d + length]
            kv = k * V + x[d + 1:d + 1 + length]
            for h in HEADS:
                SO[(h, d)] += np.bincount(pair, weights=ov[h][kv], minlength=V * V)
        print(f"{start + length:,} {time.time() - t0:.0f}s", flush=True)
    s = stats()
    P1 = np.exp(np.asarray(s.log_follow, np.float64))
    for h in HEADS:
        BO = (P1 @ ov[h].reshape(V, V).astype(np.float64).T).T
        for d in range(1, DM + 1):
            N = np.load(os.path.join(OUT, f"N_{d}.npy")).astype(np.float64)
            U = (SO[(h, d)].reshape(V, V) - N * BO) / (N + 20.0)
            np.save(os.path.join(OUT, f"UOV_h{h}_{d}.npy"), U.astype(np.float32))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
