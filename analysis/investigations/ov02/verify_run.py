"""verify_run: run analysis.vote_fit unchanged except for a memory-lean `stacked` (same values, same order).

vote_fit.stacked builds a list of per-window float16 blocks and concatenates them, so the feature stack exists twice
at its peak (the ov02 notes measured 3.7 GB RSS for this candidate). This wrapper writes each window's block straight
into one preallocated float16 array, converting float64 -> float32 -> float16 exactly as vote_fit does, so the fit
sees bit-identical inputs. Everything else (fit, evaluate, head_r2, output json) is vote_fit's own code.

Run from the repo root with the vote_fit arguments, e.g.
  OMP_NUM_THREADS=2 VOTE_FIT_THREADS=2 .venv/bin/python analysis/investigations/ov02/verify_run.py \
      analysis/candidates/ov02_earlier_votes4.py --fit_seed 2031 --eval_seed 2032 --eval 96 --tag _verify
"""
import os
import sys

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)

import analysis.vote_fit as vf  # noqa: E402


def stacked_lean(P, windows, cand, names=None):
    out, keys, beta, row = None, None, None, 0
    for w in windows:
        F, beta = vf.all_features(P, np.asarray(w[:256], dtype=np.int64), cand)
        keys = keys or list(F)
        n = np.asarray(F[keys[0]]).shape[0]
        if out is None:
            out = np.empty((len(windows) * n, vf.V, len(keys)), dtype=np.float16)
        for k_i, k in enumerate(keys):
            out[row: row + n, :, k_i] = np.asarray(F[k], np.float32)   # float32 -> float16 on assignment
        row += n
    return out[:row], keys, beta


vf.stacked = stacked_lean

if __name__ == "__main__":
    vf.main()
