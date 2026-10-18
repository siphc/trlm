"""Evaluation windows for the readable-training measurements and the program's outputs on them, computed once
(pre-registration tools b1, c, d3): data/readable/cache/<set>_windows.npy (n, 257) and <set>_prog.npy (n, 256, 2000)
float16 log-probs of program v1.3.

Sets:
  dev_fast          the harness's dev_fast (200 windows)
  train_eval        analysis.fitlib.train_windows(64, 9100)
  probe_attn_s{k}   the first 200 of attn_s{k}'s probe samples (data/eval/probe; sampled from the model at temperature 1)
  program_samples   200 windows sampled from the program at temperature 1 (token 0, then 256 sampled tokens; seed 9200)
  sent_shuffle      dev_full windows 200-399 with their sentences in a random order (seed 9201)
  word_shuffle      dev_full windows 200-399 with their words in a random order (seed 9202); a word is a token that
                    does not continue a word plus the continuation pieces after it

  .venv/bin/python -m analysis.readable_training.evalsets build [--workers 8]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from analysis.readable_training.common import REPO, T

V = 2000
CACHE = os.path.join(REPO, "data", "readable", "cache")
OOD_SETS = ["probe_twin", "program_samples", "sent_shuffle", "word_shuffle"]   # probe_twin = probe_attn_s{seed}


def path(name, kind):
    return os.path.join(CACHE, f"{name}_{kind}.npy")


def windows(name) -> np.ndarray:
    return np.load(path(name, "windows"))


def program_lp(name) -> np.ndarray:
    return np.load(path(name, "prog"), mmap_mode="r")


def _units(w, piece, boundary):
    """Split a token sequence into units: a new unit starts after a boundary token (sentences) or at every token that
    is not a word piece (words)."""
    units, cur = [], []
    for i, t in enumerate(w):
        if boundary is not None:
            cur.append(t)
            if boundary[t]:
                units.append(cur); cur = []
        else:
            if cur and not piece[t]:
                units.append(cur); cur = []
            cur.append(t)
    if cur:
        units.append(cur)
    return units


def shuffled(W, seed, by):
    from analysis.mech_eval import stats

    s = stats()
    rng = np.random.default_rng(seed)
    out = []
    for w in W:
        units = _units(list(w), s.piece > 0.5, s.sentence_end > 0.5 if by == "sentence" else None)
        order = rng.permutation(len(units))
        out.append([t for i in order for t in units[i]][: len(w)])
    return np.asarray(out, dtype=np.int64)


def sample_program(args):
    """Worker: sample windows [lo, hi) of the program samples (temperature 1)."""
    lo, hi = args
    from analysis.mech_eval import program

    P = program()
    out = []
    for k in range(lo, hi):
        rng = np.random.default_rng([9200, k])
        seq = [0]
        while len(seq) < T + 1:
            lp = P.predict(np.asarray(seq[-T:], np.int64))[-1]
            p = np.exp(lp - lp.max())
            seq.append(int(rng.choice(V, p=p / p.sum())))
        out.append(seq)
    return lo, np.asarray(out, dtype=np.int64)


def predict_set(args):
    name, lo, hi = args
    from analysis.mech_eval import program

    P = program()
    W = windows(name)
    return name, lo, np.stack([P.predict(np.asarray(w[:T], np.int64)).astype(np.float16) for w in W[lo:hi]])


def build(workers):
    import multiprocessing as mp

    from analysis.fitlib import train_windows
    from src.train.data import load_eval

    os.makedirs(CACHE, exist_ok=True)
    sets = {"dev_fast": load_eval("dev_fast").astype(np.int64), "train_eval": train_windows(64, 9100)}
    for k in range(5):
        sets[f"probe_attn_s{k}"] = np.load(os.path.join(REPO, "data", "eval", "probe", f"attn_s{k}.npy"))[:200].astype(np.int64)
    dev = load_eval("dev_full")[200:400].astype(np.int64)
    sets["sent_shuffle"] = shuffled(dev, 9201, "sentence")
    sets["word_shuffle"] = shuffled(dev, 9202, "word")
    for name, W in sets.items():
        if not os.path.exists(path(name, "windows")):
            np.save(path(name, "windows"), W)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    if not os.path.exists(path("program_samples", "windows")):
        with ctx.Pool(workers) as pool:
            parts = dict(pool.imap_unordered(sample_program, [(lo, lo + 25) for lo in range(0, 200, 25)]))
        np.save(path("program_samples", "windows"), np.concatenate([parts[lo] for lo in sorted(parts)]))
        print(f"program samples [{time.time() - t0:.0f}s]", flush=True)
    names = list(sets) + ["program_samples"]
    jobs = []
    for name in names:
        if os.path.exists(path(name, "prog")):
            continue
        n = len(windows(name))
        jobs += [(name, lo, min(lo + 25, n)) for lo in range(0, n, 25)]
    res = {}
    with ctx.Pool(workers) as pool:
        for name, lo, lp in pool.imap_unordered(predict_set, jobs):
            res.setdefault(name, {})[lo] = lp
    for name, parts in res.items():
        np.save(path(name, "prog"), np.concatenate([parts[lo] for lo in sorted(parts)]))
    print(f"program outputs for {sorted(res)} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.workers)
