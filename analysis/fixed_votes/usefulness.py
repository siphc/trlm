"""Study A, measurement M2 (logs/prereg_fixed_votes.md): the model-free usefulness of every fixed vote table, computed
from the TRAIN stream before any fixed-vote model is trained. It is the look-follows-write law's prediction.

    U_T,d(k, q) = (S - N B) / (N + 20)
    S = sum over positions j with x_j = k, x_{j+d} = q of T(k, x_{j+d+1});  B = sum_v P1(v | q) T(k, v);  N the count

as analysis/look_write/uov_tables.py (same corpus pass, chunks and shrinkage; whole stream, pairs may straddle the
end-of-story token), with the normalised tables of data/fixed_votes/tables.npz. Token-shuffled control: rows of T
permuted with default_rng(0) (S and B). A head's usefulness in logit units is beta_h * U.

    OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.fixed_votes.usefulness

Writes analysis/out/fixed_votes/usefulness/: U_<table>_d<d>.npy and Up_<table>_d<d>.npy (d = 1..7, float32 [k, q]);
U_<table>_band.npy and Up_<table>_band.npy (distances 8..63 summed, first 100M tokens); profile.json (mean usefulness
per distance d = 0..7 over all pairs, (sum S - sum N B) / sum N, and the predicted peak over d = 1..7 per table, for
test H3a); meta.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from analysis.look_write import common as C
from analysis.look_write.uov_tables import corpus_pass
from src.train.fixed_votes import TABLE_NAMES, TABLES

OUT = os.path.join(C.REPO, "analysis", "out", "fixed_votes", "usefulness")
BAND, BAND_TOKENS = (8, 63), 100_000_000


def main():
    from harness.common import load_corpus

    C.set_threads()
    C.check_disk(5.0)
    os.makedirs(OUT, exist_ok=True)
    z = np.load(TABLES)
    ov = np.stack([z[n] for n in TABLE_NAMES])                                   # (5, V, V) float32
    P1 = np.exp(z["follow"].astype(np.float64))                                  # [q, v]
    pi = C.perm()
    B = np.stack([ov[t].astype(np.float64) @ P1.T for t in range(len(TABLE_NAMES))])   # [k, q]
    Bp = B[:, pi, :]
    corpus = load_corpus()
    meta = {"tables_sha": str(z["sha"]), "tables": list(TABLE_NAMES), "shrink": C.SHRINK, "perm_seed": C.PERM_SEED,
            "band": list(BAND), "band_tokens": BAND_TOKENS, "corpus_tokens": len(corpus)}
    profile = {n: {} for n in TABLE_NAMES}
    for d in range(0, C.D_MAX + 1):
        t0 = time.time()
        S, Sp, N = corpus_pass(ov, pi, [d], corpus, log=lambda *a: None)
        for t, name in enumerate(TABLE_NAMES):
            profile[name][d] = float((S[t].sum() - (N * B[t]).sum()) / N.sum())
            if d >= 1:
                np.save(os.path.join(OUT, f"U_{name}_d{d}.npy"), ((S[t] - N * B[t]) / (N + C.SHRINK)).astype(np.float32))
                np.save(os.path.join(OUT, f"Up_{name}_d{d}.npy"),
                        ((Sp[t] - N * Bp[t]) / (N + C.SHRINK)).astype(np.float32))
        if d == 1:
            ref = np.load(os.path.join(C.REPO, "analysis", "out", "candidates", "qk", "N_1.npy"))
            meta["N_1_max_abs_diff_vs_candidates_qk"] = float(np.abs(ref.astype(np.float64) - N).max())
        print(f"d={d}: {time.time() - t0:.0f}s; mean usefulness "
              + ", ".join(f"{n} {profile[n][d]:+.4f}" for n in TABLE_NAMES), flush=True)
        del S, Sp, N
    peaks = {n: int(max(range(1, C.D_MAX + 1), key=lambda d: profile[n][d])) for n in TABLE_NAMES}
    with open(os.path.join(OUT, "profile.json"), "w") as f:
        json.dump({"mean_usefulness": profile, "predicted_peak_d1_7": peaks}, f, indent=1)
    print("predicted peaks (d = 1..7):", peaks, flush=True)
    t0 = time.time()
    S, Sp, N = corpus_pass(ov, pi, list(range(BAND[0], BAND[1] + 1)), corpus, log=lambda *a: None,
                           max_tokens=BAND_TOKENS)
    for t, name in enumerate(TABLE_NAMES):
        np.save(os.path.join(OUT, f"U_{name}_band.npy"), ((S[t] - N * B[t]) / (N + C.SHRINK)).astype(np.float32))
        np.save(os.path.join(OUT, f"Up_{name}_band.npy"), ((Sp[t] - N * Bp[t]) / (N + C.SHRINK)).astype(np.float32))
    meta["band_mean_usefulness"] = {n: float((S[t].sum() - (N * B[t]).sum()) / N.sum()) for t, n in enumerate(TABLE_NAMES)}
    print(f"band {BAND}: {time.time() - t0:.0f}s; mean usefulness {meta['band_mean_usefulness']}", flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
