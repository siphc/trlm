"""Whole-TRAIN-stream pair counts N_d[k, q]: how often token k sits exactly d tokens before token q, d = 1..8.

Corpus statistics only (no model input). Read by the law tools (analysis/look_write, look_write_deep, fixed_votes)
for pure-pair weights and as a check on their own recounts. Before the clean-up these tables lived in
analysis/out/candidates/qk/ and were written by analysis/investigations/qk/pairstats.py (git tag pre-cleanup); this
module computes the same N_d (float32, (V, V), row = earlier token k, column = later token q).

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.corpus_pairs        # about 10 min on the laptop CPU
Writes data/corpus_pairs/N_<d>.npy.
"""
from __future__ import annotations

import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, "data", "corpus_pairs")
V = 2000
DMAX = 8
CHUNK = 20_000_000


def path(d: int) -> str:
    """Path of the table N_d."""
    return os.path.join(DIR, f"N_{d}.npy")


def load(d: int) -> np.ndarray:
    """N_d as float64, (V, V) indexed [earlier token, later token]."""
    p = path(d)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing; run: .venv/bin/python -m analysis.corpus_pairs")
    return np.load(p).astype(np.float64)


def main():
    from harness.common import load_corpus

    t0 = time.time()
    corpus = load_corpus()
    n = len(corpus)
    N = {d: np.zeros(V * V, np.float64) for d in range(1, DMAX + 1)}
    for start in range(0, n - DMAX - 1, CHUNK):
        length = min(CHUNK, n - DMAX - 1 - start)
        x = np.asarray(corpus[start:start + length + DMAX + 1]).astype(np.int64)
        k = x[:length] * V
        for d in range(1, DMAX + 1):
            N[d] += np.bincount(k + x[d:d + length], minlength=V * V)
        print(f"{start + length:,} tokens, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(DIR, exist_ok=True)
    for d in range(1, DMAX + 1):
        np.save(path(d), N[d].astype(np.float32).reshape(V, V))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
