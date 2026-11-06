"""Study B, exploratory (not pre-registered; written after the confirmatory verdicts were seen).

Question: the penalised models' pure-pair correlation (the law's strength) falls to 0.06-0.21, yet the own-usefulness
arm keeps its output gain and its fitted coefficients. Does the gain come from the token main effects that the
pure-pair correlation removes by double-centring?

Per model and head, on the d = 1 type table (key k, query q; cells with N_1 >= 20, weighted by N_1, as in
analysis/look_write/fit_arms.py): split the exact pure-pair score Y and own usefulness U into an additive part
a_k + b_q (weighted least squares) and a residual. Report the weighted correlations
    raw       corr(Y, U)
    additive  corr(additive part of Y, additive part of U)
    key       corr over key tokens of the key effects a_k (weighted by the key's total count)
    pure      corr(residual of Y, residual of U)      (= the pre-registered pure-pair correlation)
Needs goal 1's usefulness tables (uov_tables.py); rebuilds them if missing and deletes them afterwards.

    OMP_NUM_THREADS=3 .venv/bin/python -m analysis.law_pressure.main_effects both_l0_s1 both_l1_s1 ...
Writes analysis/out/law_pressure/main_effects/<run>.json.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np

from analysis.look_write import common as C
from analysis.look_write.fit_arms import double_centre, wcorr
from analysis.rules_common import REPO

OUT = os.path.join(REPO, "analysis", "out", "law_pressure", "main_effects")


def additive(table, kidx, qidx, w):
    """Weighted additive fit a_k + b_q of table over the cells; returns (fitted values, key effects a_k per key)."""
    y = table[kidx, qidx].astype(np.float64)
    res = double_centre(table, kidx, qidx, w)[0]
    fit = y - res
    # key effect: weighted mean over the key's cells of the fitted values (a_k up to a constant)
    wk = np.bincount(kidx, w, C.V)
    ak = np.bincount(kidx, w * fit, C.V) / np.maximum(wk, 1e-300)
    return fit, res, ak, wk


def run(run_name):
    path = os.path.join(REPO, "models", "law_pressure", run_name, "final.pt")
    d = C.model_dir(path)
    made = False
    if not os.path.exists(os.path.join(d, "uov_h0_d1.npy")):
        subprocess.run([sys.executable, "-m", "analysis.look_write.uov_tables", path], check=True, cwd=REPO)
        made = True
    w = C.weights(C.load(path))
    Y = C.pure_pair_scores(w)
    N1 = np.load(os.path.join(d, "N_1_recount.npy")).astype(np.float64)
    kidx, qidx = np.nonzero(N1 >= C.MIN_COUNT)
    wts = N1[kidx, qidx]
    out = {}
    for h in range(C.H):
        U = np.load(os.path.join(d, f"uov_h{h}_d1.npy"))
        fy, ry, ay, wk = additive(Y[h], kidx, qidx, wts)
        fu, ru, au, _ = additive(U, kidx, qidx, wts)
        keys = wk > 0
        out[str(h)] = {"raw": wcorr(Y[h][kidx, qidx], U[kidx, qidx], wts), "additive": wcorr(fy, fu, wts),
                       "key": wcorr(ay[keys], au[keys], wk[keys]), "pure": wcorr(ry, ru, wts),
                       "share_additive_Y": float((wts * (fy - (wts * fy).sum() / wts.sum()) ** 2).sum()
                                                 / (wts * (Y[h][kidx, qidx] - (wts * Y[h][kidx, qidx]).sum()
                                                           / wts.sum()) ** 2).sum())}
        print(run_name, h, {k: round(v, 3) for k, v in out[str(h)].items()}, flush=True)
    if made:
        for f in glob.glob(os.path.join(d, "*.npy")):
            os.remove(f)
    return out


def main():
    for r in sys.argv[1:]:
        res = run(r)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{r}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
