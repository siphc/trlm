"""Corpus tables for head-3 hypotheses (TRAIN stream only; closed-form counting rules). Cached under
analysis/out/candidates/head3/.

  sub_left[s, v]   = log sum_c P(c just before s) P1(v | c) - log P(v)
                     "v can stand where s stands" (go back from s to its left neighbour c, then forward to v)
  sub_right[s, v]  = log sum_c P(c just after s) P(v just before c) - log P(v)   (same through the right neighbour)
  word[s, v]       = log P_w(v | s) - log P(v), P_w(v | s): v sits 1..4 tokens after s inside the same word (every
                     token after s up to v is a word piece), Dirichlet 20 toward P(v); 0 when s starts no piece run
                     (counts: word_counts[s, v])
"""
import os
import time

import numpy as np

from h3lib import OUT, V
from analysis.mech_eval import stats


def sub_tables(s):
    big = np.asarray(s.bigram_counts, np.float64)
    pu = np.asarray(s.p_uni, np.float64)
    p1 = np.exp(np.asarray(s.log_follow, np.float64))                    # P1(v | c)
    left = big.T / np.maximum(big.sum(0)[:, None], 1.0)                 # left[s, c] = P(c before s)
    right = big / np.maximum(big.sum(1, keepdims=True), 1.0)            # right[s, c] = P(c after s)
    before = (big + 1.0 * pu[:, None]) / (big.sum(0, keepdims=True) + 1.0)   # before[v, c] = P(v before c)
    sl = np.log(left @ p1 + 1e-12) - np.log(pu)[None]
    sr = np.log(right @ before.T + 1e-12) - np.log(pu)[None]
    return sl.astype(np.float32), sr.astype(np.float32)


def word_counts(corpus, piece):
    n = len(corpus)
    cnt = np.zeros(V * V, dtype=np.int64)
    CH = 25_000_000
    for start in range(0, n, CH):
        L = min(CH, n - start)
        seg = np.asarray(corpus[start: start + L + 4]).astype(np.int32)
        pc = piece[seg]
        run = np.ones(L, bool)
        for d in range(1, 5):
            m = min(L, len(seg) - d)
            run = run[:m] & pc[d: d + m]
            keys = seg[:m][run].astype(np.int64) * V + seg[d: d + m][run]
            cnt += np.bincount(keys, minlength=V * V)
        print("chunk", start // CH, flush=True)
    return cnt.reshape(V, V)


if __name__ == "__main__":
    s = stats()
    t0 = time.time()
    sl, sr = sub_tables(s)
    np.save(f"{OUT}/sub_left.npy", sl)
    np.save(f"{OUT}/sub_right.npy", sr)
    print("sub tables", time.time() - t0, flush=True)
    from harness.common import load_corpus
    c = word_counts(load_corpus(), np.asarray(s.piece) > 0)
    np.save(f"{OUT}/word_counts.npy", c)
    pu = np.asarray(s.p_uni, np.float64)
    pw = (c + 20 * pu[None]) / (c.sum(1, keepdims=True) + 20)
    np.save(f"{OUT}/word_pmi.npy", (np.log(pw) - np.log(pu)[None]).astype(np.float32))
    print("done", time.time() - t0)
