"""verify_causality: independent check that every feature of ov02_earlier_votes4 is causal, plus timing.

For several TRAIN windows and cut points i, replace every token after i (random tokens, all end-of-story, all word
pieces, a single-token change at i + 1) and check that rows 0..i of each candidate feature (and of the program's base
directions) are unchanged. Then time cand.features per 256-token window.

Run from the repo root: OMP_NUM_THREADS=2 .venv/bin/python analysis/investigations/ov02/verify_causality.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import time

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)

from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from analysis.vote_fit import base_features, load_candidate  # noqa: E402

CAND = os.path.join(REPO, "analysis", "candidates", "ov02_earlier_votes4.py")


def main():
    P = program()
    cand = load_candidate(CAND)
    t0 = time.time()
    cand.features(P, np.arange(8, dtype=np.int64))
    print(f"first call (table load) {time.time() - t0:.2f}s")
    rng = np.random.default_rng(31337)
    W = train_windows(4, seed=4242)
    shape = np.asarray(P.STATS.shape)
    pieces = np.flatnonzero(shape == 1)
    worst = {}
    n_checks = 0
    for w in W:
        t = np.asarray(w[:256], dtype=np.int64)
        F = cand.features(P, t)
        B, _ = base_features(P, t)
        for i in (0, 1, 2, 7, 8, 9, 63, 128, 200, 254):
            for mode in ("random", "eos", "pieces", "single"):
                t2 = t.copy()
                m = 255 - i
                if mode == "random":
                    t2[i + 1:] = rng.integers(0, 2000, m)
                elif mode == "eos":
                    t2[i + 1:] = 0
                elif mode == "pieces":
                    t2[i + 1:] = rng.choice(pieces, m)
                else:
                    t2[i + 1] = (t2[i + 1] + 1 + rng.integers(0, 1999)) % 2000
                assert (t2[: i + 1] == t[: i + 1]).all() and (t2 != t).any()
                F2 = cand.features(P, t2)
                B2, _ = base_features(P, t2)
                for name, (a, b) in list({f"cand.{k}": (F[k], F2[k]) for k in F}.items()) + \
                        list({f"base.{k}": (B[k], B2[k]) for k in B}.items()):
                    a, b = np.asarray(a), np.asarray(b)
                    d = float(np.abs(a[: i + 1] - b[: i + 1]).max())
                    worst[name] = max(worst.get(name, 0.0), d)
                    # the perturbation must actually reach later rows (sanity: the check is not vacuous)
                    if name.startswith("cand.") and mode == "random":
                        worst.setdefault(name + "|later_changed", 0.0)
                        worst[name + "|later_changed"] = max(worst[name + "|later_changed"],
                                                             float(np.abs(a[i + 1:] - b[i + 1:]).max()))
                n_checks += 1
    print(f"{n_checks} perturbations x {len([k for k in worst if '|' not in k])} features")
    for k in sorted(worst):
        print(f"  {k:45s} max |diff| = {worst[k]:.3e}")
    bad = [k for k in worst if "|" not in k and worst[k] > 0.0]
    print("CAUSAL (exact):", not bad, bad)

    # timing per 256-token window, steady state
    ts = []
    for w in train_windows(8, seed=777):
        t = np.asarray(w[:256], dtype=np.int64)
        t1 = time.perf_counter()
        cand.features(P, t)
        ts.append(time.perf_counter() - t1)
    tb = []
    for w in train_windows(4, seed=778):
        t = np.asarray(w[:256], dtype=np.int64)
        t1 = time.perf_counter()
        base_features(P, t)
        tb.append(time.perf_counter() - t1)
    print(f"candidate features per window: mean {np.mean(ts):.3f}s median {np.median(ts):.3f}s (n={len(ts)}); "
          f"base features (all 25 dirs) mean {np.mean(tb):.3f}s")


if __name__ == "__main__":
    main()
