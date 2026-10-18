"""Dose-response table of the development runs (models/readable/dev/*): dev_full cross-entropy and its cost against the
lambda = 0 control, and the online metrics at the last evaluation (src/train/readable.py ReadableAux.metrics).

    .venv/bin/python -m analysis.readable_training.grid_table [--root models/readable/dev]
Writes analysis/out/readable_training/grid_table.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os

from analysis.readable_training.common import OUT, REPO

COLS = [("m_kl_model_prog", "KL(model‖prog)"), ("m_kl_prog_model", "KL(prog‖model)"),
        ("m_att_kl_model_rule", "attn KL(model‖rule)"), ("m_att_kl_rule_model", "attn KL(rule‖model)"),
        ("vote_uniform", "vote ρ (uniform, mean h0-2)"),
        ("gap_out", "memo gap out"), ("gap_att", "memo gap attn")]


def last_metrics(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    m = [r for r in rows if "m_kl_model_prog" in r]
    if not m:
        return {}
    x = dict(m[-1])
    x["vote_uniform"] = sum(x[f"m_vote_rho_uniform_h{h}"] for h in range(3)) / 3
    x["gap_out"] = x["m_heldout_out_coarse"] - x["m_train_out_coarse"]
    x["gap_att"] = x["m_heldout_att_coarse"] - x["m_train_att_coarse"]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "models", "readable", "dev"))
    a = ap.parse_args()
    runs = {}
    for name in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, name, "result.json")
        if os.path.exists(p):
            runs[name] = {"dev_full": json.load(open(p))["dev_full_loss"], **last_metrics(os.path.join(a.root, name))}
    base = runs.get("none_s0_100M", {}).get("dev_full")
    lines = ["| run | dev_full CE | cost | " + " | ".join(c for _, c in COLS) + " |",
             "|---|---|---|" + "---|" * len(COLS)]
    for name, r in runs.items():
        cost = f"{r['dev_full'] / base - 1:+.2%}" if base else ""
        lines.append(f"| {name} | {r['dev_full']:.4f} | {cost} | "
                     + " | ".join(f"{r[k]:.3f}" if k in r else "" for k, _ in COLS) + " |")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "grid_table.json"), "w") as f:
        json.dump(runs, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
