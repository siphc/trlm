"""Apply the decision rules R1-R3 (logs/deep_study.md, fixed before any calibration output) to the one-layer
calibration (200M-token passes; arms_ltotal.json and arms_total.json of attn_s1..s4, the 8 confirmatory heads).

  R1  primary readout: ltotal, unless total's median own-arm gain exceeds ltotal's by more than 10%
  R2  theta: 0.5, unless the median own/oracle gain ratio (primary readout) is below 0.75; then 2/3 x that median,
      rounded down to a multiple of 0.05
  R3  D5 references: medians of the own-arm coefficients c1, c2 (primary readout)

    .venv/bin/python -m analysis.look_write_deep.apply_rules
Writes analysis/out/look_write_deep/rules.json.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from analysis.look_write_deep import common_deep as C

MODELS = ["attn_s1", "attn_s2", "attn_s3", "attn_s4"]


def heads(model, readout):
    a = json.load(open(os.path.join(C.model_dir(model), f"arms_{readout}.json")))
    assert a["n_tokens"] >= 199_000_000, (model, a["n_tokens"])
    return {k: v for k, v in a["heads"].items() if v["class"]["confirmatory"]}


def main():
    gains = {r: [] for r in ("ltotal", "total")}
    for m in MODELS:
        for r in gains:
            gains[r] += [hd["arms"][f"own_{r}"]["gain"] for hd in heads(m, r).values()]
    med = {r: float(np.median(v)) for r, v in gains.items()}
    primary = "total" if med["total"] > 1.10 * med["ltotal"] else "ltotal"
    ratios, c1, c2 = [], [], []
    for m in MODELS:
        for hd in heads(m, primary).values():
            own, oqc = hd["arms"][f"own_{primary}"], hd["arms"]["oqc"]
            ratios.append(own["gain"] / oqc["gain"] if oqc["gain"] > 0 else float("nan"))
            c1.append(own["coef"]["c1"])
            c2.append(own["coef"]["c2"])
    med_ratio = float(np.nanmedian(ratios))
    theta = 0.5 if med_ratio >= 0.75 else math.floor((2 / 3) * med_ratio / 0.05) * 0.05
    out = {"R1": {"median_own_gain": med, "primary": primary},
           "R2": {"ratios": ratios, "median_ratio": med_ratio, "theta": round(theta, 2)},
           "R3": {"c1_median": float(np.median(c1)), "c2_median": float(np.median(c2)), "c1": c1, "c2": c2},
           "n_heads": len(gains["ltotal"])}
    with open(os.path.join(C.OUT, "rules.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"primary": primary, "median_own_gain": med, "median_ratio": med_ratio, "theta": out["R2"]["theta"],
                      "c1": out["R3"]["c1_median"], "c2": out["R3"]["c2_median"], "n_heads": out["n_heads"]}, indent=1))


if __name__ == "__main__":
    main()
