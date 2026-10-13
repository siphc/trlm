"""Verifier: causality and cost of the candidate analysis/candidates/recall_attn_pmi1_votes_b25.py.

For several TRAIN windows and cut points i, replace every token after position i (random tokens, and a second
perturbation that shifts them) and check that rows <= i of every feature the candidate arm of vote_fit uses (the
candidate's attention matrix, its 20 vote directions, and the remaining base directions) are unchanged.
Then time cand.features per 256-token window.

  OMP_NUM_THREADS=2 .venv/bin/python -m analysis.investigations.recall.verify_causal_pmi1_b25
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "investigations", "recall"))

from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from analysis.vote_fit import all_features, load_candidate  # noqa: E402
import cand_common as C  # noqa: E402

CAND = os.path.join(REPO, "analysis", "candidates", "recall_attn_pmi1_votes_b25.py")


def feats(P, cand, t):
    F, _ = all_features(P, t, cand)
    A = C.attention(P, t, "pmi1")
    out = {k: np.asarray(v, np.float64) for k, v in F.items()}
    out["__attention_pmi1"] = np.asarray(A, np.float64)
    return out


def main():
    P = program()
    cand = load_candidate(CAND)
    rng = np.random.default_rng(7)
    W = train_windows(4, seed=2033)
    cuts = [0, 1, 37, 128, 200, 254]
    worst = {}
    n_checks = 0
    failures = []
    for wi, w in enumerate(W):
        t = np.asarray(w[:256], np.int64)
        f0 = feats(P, cand, t)
        for i in cuts:
            for kind in ("random", "shift"):
                t2 = t.copy()
                if kind == "random":
                    t2[i + 1:] = rng.integers(0, 2000, size=255 - i)
                else:
                    t2[i + 1:] = (t[i + 1:] + 1 + rng.integers(0, 1999)) % 2000
                assert (t2[: i + 1] == t[: i + 1]).all() and (i == 255 or (t2[i + 1:] != t[i + 1:]).any())
                f1 = feats(P, cand, t2)
                for k in f0:
                    d = float(np.max(np.abs(f0[k][: i + 1] - f1[k][: i + 1])))
                    worst[k] = max(worst.get(k, 0.0), d)
                    n_checks += 1
                    if d > 1e-9:
                        failures.append((wi, i, kind, k, d))
                # sanity: the future rows should change for the attention (the test can detect a change)
                if i < 254:
                    assert np.max(np.abs(f0["__attention_pmi1"][i + 1:] - f1["__attention_pmi1"][i + 1:])) > 0
    cand_keys = [k for k in worst if k.startswith(cand.NAME + ".")] + ["__attention_pmi1"]
    print(json.dumps({"checks": n_checks, "failures": failures[:20], "n_failures": len(failures),
                      "max_abs_diff_candidate_features": {k: worst[k] for k in cand_keys},
                      "max_abs_diff_any_feature": max(worst.values())}, indent=1))

    # timing: candidate features per 256-token window (after warm-up, tables cached)
    Wt = train_windows(12, seed=2034)
    cand.features(P, np.asarray(Wt[0][:256], np.int64))
    ts = []
    for w in Wt[1:]:
        t = np.asarray(w[:256], np.int64)
        t0 = time.perf_counter()
        cand.features(P, t)
        ts.append(time.perf_counter() - t0)
    ta = []
    for w in Wt[1:5]:
        t = np.asarray(w[:256], np.int64)
        t0 = time.perf_counter()
        C.attention(P, t, "pmi1")
        ta.append(time.perf_counter() - t0)
    print(json.dumps({"seconds_per_window_candidate_features": {"mean": float(np.mean(ts)), "median": float(np.median(ts)),
                                                                "max": float(np.max(ts)), "n": len(ts)},
                      "seconds_per_window_attention_only": float(np.mean(ta))}, indent=1))


if __name__ == "__main__":
    main()
