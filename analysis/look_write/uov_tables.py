"""Corpus pass: usefulness tables UOV_h,d(k, q) for every head and d = 1..7, plus the token-shuffled control.

    S[k, q]  = sum over TRAIN positions j with x_j = k, x_{j+d} = q of OV_h(k, x_{j+d+1})
    B[k, q]  = sum_v P1(v | q) OV_h(k, v)
    UOV      = (S - N B) / (N + 20)

The control replaces OV_h by OV_h[perm] (rows permuted over k with default_rng(0)) everywhere (S and B).
N is recounted in the same pass over exactly the same positions (j = 0 .. L-d-2) and compared with
analysis/out/candidates/qk/N_d.npy. One weighted np.bincount per (head, d, chunk), chunks of 10M positions.
All positions of the stream are used (pairs may straddle the end-of-story token 0, as attention does).

Writes analysis/out/look_write/<model>/{uov,uovp}_h<h>_d<d>.npy (float32, [k, q]) and meta.json.
Run: OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write.uov_tables attn_s1 attn_s2 ...
     (--band 8 63 [--max-tokens 100000000] for the exploratory distance-band version: one table per head,
      uovband_h<h>.npy / uovpband_h<h>.npy, S and N summed over d = 8..63)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from analysis.look_write import common as C


def corpus_pass(ov, pi, ds, corpus, log=print, max_tokens=None):
    """Accumulate S (own), Sp (permuted) for each head, summed over distances ds, and N. Returns float64 arrays."""
    V, Hn = C.V, ov.shape[0]
    L = len(corpus) if max_tokens is None else min(len(corpus), max_tokens)
    ovf = [ov[h].ravel() for h in range(Hn)]
    S = np.zeros((Hn, V * V))
    Sp = np.zeros((Hn, V * V))
    N = np.zeros(V * V)
    for d in ds:
        t0 = time.time()
        jmax = L - d - 1                      # j = 0 .. L-d-2 so that x_{j+d+1} exists
        for a in range(0, jmax, C.CHUNK):
            b = min(a + C.CHUNK, jmax)
            x = np.asarray(corpus[a:b + d + 1]).astype(np.int64)
            m = b - a
            kk = x[:m]
            key = kk * V + x[d:d + m]
            nxt = x[d + 1:d + 1 + m]
            iw = kk * V + nxt
            iwp = pi[kk] * V + nxt
            N += np.bincount(key, minlength=V * V)
            for h in range(Hn):
                S[h] += np.bincount(key, weights=ovf[h][iw], minlength=V * V)
                Sp[h] += np.bincount(key, weights=ovf[h][iwp], minlength=V * V)
            del x, kk, key, nxt, iw, iwp
        log(f"  d={d} done in {time.time() - t0:.0f}s")
    return S.reshape(Hn, V, V), Sp.reshape(Hn, V, V), N.reshape(V, V)


def model_setup(model_id):
    m = C.load(model_id)
    w = C.weights(m)
    scale = C.final_ln_scale(m, C.train_windows(C.FIT_SEED))
    ov = C.ov_tables(w, scale)
    return m, w, scale, ov


def build(model_id, band=None, max_tokens=None):
    from harness.common import load_corpus

    C.check_disk()
    out = C.model_dir(model_id)
    m, w, scale, ov = model_setup(model_id)
    if w["mlp"]:
        print(f"{model_id}: MLP model, OV through the direct path only (exploratory)")
    P1 = C.p1_table()
    pi = C.perm()
    corpus = load_corpus()
    print(f"{model_id}: final-LN scale {scale:.4f}", flush=True)
    meta_path = os.path.join(out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"model": model_id, "final_ln_scale": scale, "perm_seed": C.PERM_SEED, "shrink": C.SHRINK})
    B = np.stack([ov[h].astype(np.float64) @ P1.T for h in range(C.H)])          # [k, q]
    Bp = B[:, pi, :]                                                              # B with OV rows permuted
    if band is None:
        ncheck = {}
        for d in range(1, C.D_MAX + 1):
            S, Sp, N = corpus_pass(ov, pi, [d], corpus)
            ref = np.load(os.path.join(C.REPO, "analysis", "out", "candidates", "qk", f"N_{d}.npy"))
            diff = np.abs(ref.astype(np.float64) - N)
            ncheck[d] = {"max_abs_diff": float(diff.max()), "cells_diff_gt_0.5": int((diff > 0.5).sum()),
                         "max_rel_diff_cells_ge_1e6": float((diff / np.maximum(N, 1))[N >= 1e6].max(initial=0))}
            for h in range(C.H):
                np.save(os.path.join(out, f"uov_h{h}_d{d}.npy"),
                        ((S[h] - N * B[h]) / (N + C.SHRINK)).astype(np.float32))
                np.save(os.path.join(out, f"uovp_h{h}_d{d}.npy"),
                        ((Sp[h] - N * Bp[h]) / (N + C.SHRINK)).astype(np.float32))
            if d == 1:
                np.save(os.path.join(out, "N_1_recount.npy"), N.astype(np.float32))
            print(f"  d={d}: N check {ncheck[d]}", flush=True)
            del S, Sp, N
        meta["N_check_vs_candidates_qk"] = ncheck
    else:
        lo, hi = band
        S, Sp, N = corpus_pass(ov, pi, list(range(lo, hi + 1)), corpus, max_tokens=max_tokens)
        for h in range(C.H):
            np.save(os.path.join(out, f"uovband_h{h}.npy"), ((S[h] - N * B[h]) / (N + C.SHRINK)).astype(np.float32))
            np.save(os.path.join(out, f"uovpband_h{h}.npy"),
                    ((Sp[h] - N * Bp[h]) / (N + C.SHRINK)).astype(np.float32))
        meta["band"] = [lo, hi]
        meta["band_max_tokens"] = max_tokens
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--band", nargs=2, type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="band mode only: use the first N tokens")
    a = ap.parse_args()
    C.set_threads()
    for mid in a.models:
        t0 = time.time()
        build(mid, a.band, a.max_tokens)
        print(f"{mid}: tables built in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
