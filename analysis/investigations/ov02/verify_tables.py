"""verify_tables: independently recount the two cached tables ov02_earlier_votes4 loads, from the TRAIN stream.

  KPRED(s, v) = sum_a sqrt(P(prev = a | s) P(prev = a | v)), from the program's whole-stream bigram counts
  WC_3(s, v)  = # positions j with t_j = s whose 3rd unit after the unit of j starts with v, no end-of-story token
                strictly between (a unit starts at every token whose shape is not 'continuation')
  WPMI_3      = Dirichlet + background smoothed PMI of WC_3, constants from program/corpus.py

The recount uses a different algorithm from s03_word_tables.py (searchsorted over unit starts and over end-of-story
positions instead of cumulative sums), with the same 25M-token chunk boundaries so the counts can be compared exactly.

Run from the repo root: OMP_NUM_THREADS=2 .venv/bin/python analysis/investigations/ov02/verify_tables.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import time

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)

from analysis.mech_eval import stats  # noqa: E402
from harness.common import CORPUS, load_corpus  # noqa: E402
from program.corpus import BACKGROUND, CHUNK, DIRICHLET  # noqa: E402

OUT = os.path.join(REPO, "analysis", "out", "candidates", "ov02")
V = 2000
K = 3
MARGIN = 400
BLOCK = 5_000_000


def recount_wc3(corpus, shape):
    n = len(corpus)
    counts = np.zeros(V * V, dtype=np.int64)
    dropped_no_unit = dropped_tail = 0
    for start in range(0, n, CHUNK):
        length = min(CHUNK, n - start)
        seg = np.asarray(corpus[start: start + length + MARGIN]).astype(np.int32)
        starts = np.flatnonzero(shape[seg] != 1).astype(np.int64)
        eos = np.flatnonzero(seg == 0).astype(np.int64)
        for b in range(0, length, BLOCK):
            j = np.arange(b, min(b + BLOCK, length), dtype=np.int64)
            u = np.searchsorted(starts, j, side="right") - 1          # index of the unit containing j
            ok = u >= 0
            dropped_no_unit += int((~ok).sum())
            ok &= (u + K) < len(starts)
            dropped_tail += int(((u >= 0) & ((u + K) >= len(starts))).sum())
            j, u = j[ok], u[ok]
            tp = starts[u + K]
            k = np.searchsorted(eos, j, side="right")                 # first end-of-story strictly after j
            nxt = np.where(k < len(eos), eos[np.minimum(k, len(eos) - 1)], np.iinfo(np.int64).max)
            keep = nxt >= tp                                           # none strictly between j and tp
            counts += np.bincount(seg[j[keep]].astype(np.int64) * V + seg[tp[keep]], minlength=V * V)
        print(f"  chunk {start // CHUNK}", flush=True)
    return counts.reshape(V, V), dropped_no_unit, dropped_tail


def main():
    t0 = time.time()
    S = stats()
    print("corpus file:", CORPUS)
    pu = np.asarray(S.p_uni, np.float64)
    shape = np.asarray(S.shape)

    big = np.asarray(S.bigram_counts, np.float64)
    rp = np.sqrt(big / np.maximum(big.sum(0, keepdims=True), 1))
    kp = rp.T @ rp
    kc = np.load(os.path.join(OUT, "kpred.npy"))
    print(f"KPRED recomputed vs cached: max |diff| {np.abs(kp - kc).max():.2e} (float32 cache), "
          f"diag mean {np.diag(kp).mean():.3f}")

    corpus = load_corpus()
    print(f"corpus tokens {len(corpus):,}; program stats n_tokens {int(S.n_tokens):,}")
    t1 = time.time()
    wc3, d0, d1 = recount_wc3(corpus, shape)
    print(f"WC_3 recount {time.time() - t1:.0f}s; positions dropped: before first unit of a chunk {d0}, "
          f"no 3rd unit within margin {d1}")
    cached = np.load(os.path.join(OUT, "wc.npy"), mmap_mode="r")[3]
    diff = np.abs(wc3 - cached.astype(np.int64))
    print(f"WC_3 recount vs cached wc.npy[3]: total {int(wc3.sum()):,} vs {int(cached.sum()):,}; "
          f"cells differing {int((diff > 0).sum())}, max |diff| {int(diff.max())}")
    c = wc3.astype(np.float64)
    p = (1 - BACKGROUND) * (c + DIRICHLET * pu) / (c.sum(1, keepdims=True) + DIRICHLET) + BACKGROUND * pu
    w3 = np.log(p) - np.log(pu)
    w3c = np.load(os.path.join(OUT, "wpmi_3.npy"))
    print(f"WPMI_3 from recount vs cached wpmi_3.npy: max |diff| {np.abs(w3 - w3c).max():.2e}")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
