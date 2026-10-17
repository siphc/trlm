"""Sensitivity (not part of the pre-registration): bootstrap the 64 eval windows (same resample for every head of a
model) and recompute D1, D2 (D3 is window-independent) and the family verdicts.

    .venv/bin/python analysis/out/look_write_deep/verify/bootstrap_verdicts.py
"""
import json
from collections import Counter

import numpy as np

OUT = "/Users/svi/Desktop/trlm/analysis/out/look_write_deep"
MODELS = [f"attn2L_s{s}" for s in range(1, 5)] + [f"attn4L_s{s}" for s in range(1, 5)]
M = json.load(open(f"{OUT}/verify/verdicts_verify.json"))
rng = np.random.default_rng(0)
data = {}
for m in MODELS:
    A = json.load(open(f"{OUT}/{m}/arms_ltotal.json"))
    for hk, rec in A["heads"].items():
        key = f"{m}/{hk}"
        if key in M["heads"]:
            arms = rec["arms"]
            kw = {a: np.asarray(v["kl_win"]) for a, v in arms.items()}
            data[key] = (m, kw, rec["near_duplicates"], M["heads"][key])
counts = {f: Counter() for f in ("F1", "F2", "F3")}
n_boot = 2000
for _ in range(n_boot):
    idx = {m: rng.integers(0, 64, 64) for m in MODELS}
    res = {}
    for key, (m, kw, dups, v) in data.items():
        i = idx[m]
        base = kw["base"][i].mean()
        g = {a: base - w[i].mean() for a, w in kw.items()}
        se = (kw["base"][i] - kw["oqc"][i]).std(ddof=1) / 8
        ev = g["oqc"] >= max(0.01, 3 * se)
        d1 = g["own_ltotal"] >= 0.2 * g["oqc"] if ev else g["own_ltotal"] >= 0.15 * base
        ctrl = max([g[a] for a in g if a.startswith("other_") and a not in dups] + [g["shuffled"]])
        d2 = g["own_ltotal"] > 0 and ctrl <= 0.25 * g["own_ltotal"]
        res[key] = (v["family"], d1, d1 and d2 and v["D3"])
    for f in counts:
        hs = [r for r in res.values() if r[0] == f]
        n = len(hs)
        n123 = sum(r[2] for r in hs)
        nf = sum(not r[1] for r in hs)
        counts[f]["SUPPORTED" if n123 >= 0.8 * n else "REFUTED" if nf >= 0.5 * n else "MIXED"] += 1
for f, c in counts.items():
    print(f, {k: round(v / n_boot, 3) for k, v in c.items()})
