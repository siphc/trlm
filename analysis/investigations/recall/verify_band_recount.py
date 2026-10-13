"""Verifier: recount the band-25..64 table from the TRAIN corpus by its stated closed-form rule and compare it with
the cached table the candidate actually loads (analysis/scratch/head1_card/PMI_25_64.npy).

Rule (program/corpus.py topic_table with distances 25..64): over the first 50M TRAIN tokens, count (s, v) for every
position m+d (d in 25..64) whose token is v while the most recent occurrence of s at or before m+d-1 is at m
(no other s in between); slots[s] = number of such (m, d); P(v) = (uni + 1) / sum(uni + 1) on those 50M tokens;
band(s, v) = log((count + 100 P(v)) / (slots + 100)) - log P(v). Memory-lean (int32) but the same counts.
"""
import os
import sys
import time

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
from harness.common import load_corpus  # noqa: E402

N, V, PRIOR = 50_000_000, 2000, 100.0
t0 = time.time()
x = np.asarray(load_corpus()[:N]).astype(np.int32)
n = len(x)
order = np.argsort(x, kind="stable")
xs = x[order]
gap_sorted = np.full(n, 10 ** 9, dtype=np.int32)
same = xs[1:] == xs[:-1]
gap_sorted[:-1] = np.where(same, (order[1:] - order[:-1]).astype(np.int32), 10 ** 9)
del xs, same
gap = np.empty(n, dtype=np.int32)
gap[order] = gap_sorted
del order, gap_sorted
print("gaps", round(time.time() - t0, 1), flush=True)
counts = np.zeros(V * V, dtype=np.int64)
slots = np.zeros(V, dtype=np.int64)
for d in range(25, 65):
    keep = gap[: n - d] >= d
    src, dst = x[: n - d][keep], x[d:][keep]
    counts += np.bincount(src * V + dst, minlength=V * V)
    slots += np.bincount(src, minlength=V)
del gap
uni = np.bincount(x, minlength=V).astype(np.float64)
pv = (uni + 1.0) / (uni + 1.0).sum()
c = counts.reshape(V, V).astype(np.float64)
table = (np.log((c + PRIOR * pv[None, :]) / (slots[:, None] + PRIOR)) - np.log(pv)[None, :]).astype(np.float32)
cached = np.load(os.path.join(REPO, "analysis", "scratch", "head1_card", "PMI_25_64.npy"))
d = np.abs(table.astype(np.float64) - cached.astype(np.float64))
print({"seconds": round(time.time() - t0, 1), "shape": cached.shape, "max_abs_diff": float(d.max()),
       "mean_abs_diff": float(d.mean()), "n_diff_gt_1e-4": int((d > 1e-4).sum())})
