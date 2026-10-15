"""Head classes by fixed rules, computed before any outcome (pre-registration, deep study).

  LOCAL      pooled mean attention distance to earlier tokens <= 10 (fit windows, as in the one-layer study)
  PREV       mean attention on the previous token >= 0.5
  INDUCTION  on 32 synthetic windows (128 random non-EOS tokens, repeated twice), mean attention from positions in the
             second copy to the token after the earlier occurrence of the current token >= 0.3
  confirmatory = LOCAL and not PREV and not INDUCTION and earlier-token mass >= 0.15

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.classify attn2L_s0 ...
Writes analysis/out/look_write_deep/<model>/classes.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from analysis.look_write.fit_arms import mean_distances
from analysis.look_write_deep import common_deep as C

PREV_MIN, INDUCTION_MIN, EARLIER_MASS_MIN, LOCAL_MAX = 0.5, 0.3, 0.15, 10.0


def induction_scores(model, n=32, half=128, seed=0):
    rng = np.random.default_rng(seed)
    seq = rng.integers(1, C.V, size=(n, half))
    wins = np.concatenate([seq, seq], 1)
    pats = C.patterns(model, wins)
    i = np.arange(half + 1, 2 * half)                 # query positions in the second copy (need i - half + 1 < i)
    out = []
    for p in pats:                                    # p: (n, H, 2*half, 2*half)
        out.append(p[:, :, i, i - half + 1].mean(axis=(0, 2)))
    return out


def run(model_id):
    m = C.load(model_id)
    L, H, _, _ = C.dims(m)
    fit = C.windows(C.FIT_SEED, C.N_FIT)[:, :C.T]
    pats = C.patterns(m, fit)
    ind = induction_scores(m)
    heads = []
    for l in range(L):
        md = mean_distances(pats[l])
        prev = pats[l][:, :, np.arange(1, C.T), np.arange(0, C.T - 1)].mean(axis=(0, 2))
        for h in range(H):
            row = {"layer": l, "head": h, "mean_distance": md[h]["pooled"], "earlier_mass": md[h]["earlier_mass"],
                   "prev_attention": float(prev[h]), "induction_score": float(ind[l][h])}
            row["LOCAL"] = row["mean_distance"] <= LOCAL_MAX
            row["PREV"] = row["prev_attention"] >= PREV_MIN
            row["INDUCTION"] = row["induction_score"] >= INDUCTION_MIN
            row["confirmatory"] = (row["LOCAL"] and not row["PREV"] and not row["INDUCTION"]
                                   and row["earlier_mass"] >= EARLIER_MASS_MIN)
            heads.append(row)
    res = {"model": model_id, "n_blocks": L, "rules": {"LOCAL_MAX": LOCAL_MAX, "PREV_MIN": PREV_MIN,
                                                        "INDUCTION_MIN": INDUCTION_MIN,
                                                        "EARLIER_MASS_MIN": EARLIER_MASS_MIN}, "heads": heads}
    with open(f"{C.model_dir(model_id)}/classes.json", "w") as f:
        json.dump(res, f, indent=1)
    for r in heads:
        print(f"{model_id} L{r['layer']}H{r['head']}: dist {r['mean_distance']:.1f} earlier {r['earlier_mass']:.2f} "
              f"prev {r['prev_attention']:.2f} induction {r['induction_score']:.2f} -> "
              f"{'CONFIRMATORY' if r['confirmatory'] else ','.join(k for k in ('LOCAL', 'PREV', 'INDUCTION') if r[k]) or 'RECALL'}",
              flush=True)
    return res


if __name__ == "__main__":
    torch.set_num_threads(2)
    for mid in sys.argv[1:]:
        run(mid)
