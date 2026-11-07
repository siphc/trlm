"""Drivers for the ladder of tiny language models (logs/ladder/design.md, decision 16; configs/side_ladder.yaml).

Every run is resumable and skipped when its result.json exists.

  python -m src.train.ladder_runs counts                     # parameter counts of every rung -> logs/ladder/param_counts.json
  python -m src.train.ladder_runs pilot_lr --rung L1         # seed 0, arm ln, smallest budget, LR in PILOT_LRS
  python -m src.train.ladder_runs pilot_budget --rung L1     # seed 0, arm ln, budgets PILOT_BUDGETS at the best LR
  python -m src.train.ladder_runs pilot_report               # writes logs/ladder/pilot.md
  python -m src.train.ladder_runs rung --rung L1 --arm ln --seeds 0,1,2,3,4
                                                             # models/ladder/{rung}_{arm}_s{seed}
"""
from __future__ import annotations

import argparse
import json
import os

from src.train.train import TrainConfig, train

# The rungs (design, Models). One change per step. L9 (BPE 256) and L10 (the existing attn_s0..s4) are not trained here.
RUNGS = {
    "L1": dict(d_model=4, n_heads=1, context_length=64),
    "L2": dict(d_model=8, n_heads=1, context_length=64),
    "L3": dict(d_model=8, n_heads=2, context_length=64),
    "L4": dict(d_model=16, n_heads=2, context_length=64),
    "L5": dict(d_model=32, n_heads=4, context_length=64),
    "L6": dict(d_model=64, n_heads=4, context_length=64),
    "L7": dict(d_model=128, n_heads=4, context_length=64),
    "L8": dict(d_model=128, n_heads=4, context_length=256),
}
ARMS = {"ln": True, "noln": False}
LN_ARM_RUNGS = ("L1", "L2", "L3")         # rungs trained in both LayerNorm arms (design, Models)
PILOT_RUNGS = ("L1", "L5")                # the pilot's rungs; the budget applies to the rungs next to each
PILOT_LRS = [1e-3, 3e-3, 1e-2]
# If the best LR is the largest in the grid, the next step up is also run at the smallest budget (and so on while it
# keeps winning); best_lr() then picks from every LR run. Added 2026-11-06 when both pilot rungs chose 1e-2.
PILOT_LR_STEP = 3.0
PILOT_BUDGETS = [25_000_000, 100_000_000, 400_000_000]
WITHIN = 0.01                             # choose the smallest budget within 1% of the best dev loss
EPOCH_SYMBOLS = 1_884_121_742             # data/char/meta.json n_symbols.train
# If the largest budget beats the next smaller one by more than WITHIN, one more step (x4, capped at one epoch) is run,
# as the main-track pilot's cap at one epoch (AGENTS.md section 5).
PILOT_EXTRA = (EPOCH_SYMBOLS - 1) // 64 // 64 * 64 * 64
PILOT_ROOT = "models/ladder_pilot"
ROOT = "models/ladder"
# Filled in from logs/ladder/pilot.md after the pilot (rung -> (peak LR, budget in symbols)); empty until then.
CHOSEN: dict[str, tuple[float, int]] = {}


def config(rung: str, arm: str, seed: int, peak_lr: float, budget: int, out: str, device: str) -> TrainConfig:
    return TrainConfig(variant="attn", seed=seed, data="char", layernorm=ARMS[arm], budget_tokens=budget,
                       peak_lr=peak_lr, out=out, device=device, n_log_checkpoints=10, log_every=200,
                       eval_every=2000, **RUNGS[rung])


def run(cfg: TrainConfig) -> dict:
    res = os.path.join(cfg.out, "result.json")
    if os.path.exists(res):
        with open(res) as f:
            return json.load(f)
    print(f"=== {cfg.out}", flush=True)
    return train(cfg)


def lr_name(lr: float) -> str:
    return f"{lr:.0e}".replace("-0", "-")


def pilot_dir(rung: str, lr: float, budget: int) -> str:
    return f"{PILOT_ROOT}/{rung}_ln_lr{lr_name(lr)}_b{budget // 1_000_000}M"


def lrs_run(rung: str) -> list[float]:
    """The pilot LRs of a rung: the grid plus any edge steps already run."""
    out = list(PILOT_LRS)
    while os.path.exists(f"{pilot_dir(rung, round(out[-1] * PILOT_LR_STEP, 6), PILOT_BUDGETS[0])}/result.json"):
        out.append(round(out[-1] * PILOT_LR_STEP, 6))
    return out


def best_lr(rung: str) -> float:
    rows = [(json.load(open(f"{pilot_dir(rung, lr, PILOT_BUDGETS[0])}/result.json"))["dev_full_loss"], lr)
            for lr in lrs_run(rung)]
    return min(rows)[1]


def counts() -> dict:
    from src.train.model import TinyLM, count_params
    from src.train.train import ladder_model_config

    out = {}
    for rung in RUNGS:
        for arm in ARMS:
            m = TinyLM(ladder_model_config(config(rung, arm, 0, 1e-3, 1, "x", "cpu")))
            out[f"{rung}_{arm}"] = count_params(m)
    os.makedirs("logs/ladder", exist_ok=True)
    with open("logs/ladder/param_counts.json", "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, v["total"])
    return out


def pilot_report() -> None:
    lines = ["# Ladder pilot (logs/ladder/design.md, Training)", "",
             "Seed 0, arm ln. LR check at the smallest budget; budget ladder at the best LR, each run with its own "
             "full cosine schedule. Dev loss is the mean cross-entropy on dev_full (nats per symbol).", ""]
    for rung in PILOT_RUNGS:
        lines += [f"## {rung}", "", "| peak LR | budget (symbols) | dev_full loss | steps | seconds |", "|---|---|---|---|---|"]
        rows = [(lr, PILOT_BUDGETS[0]) for lr in lrs_run(rung)]
        try:
            lr_b = best_lr(rung)
        except FileNotFoundError:
            lr_b = None
        if lr_b is not None:
            rows += [(lr_b, b) for b in PILOT_BUDGETS[1:]]
            if os.path.exists(f"{pilot_dir(rung, lr_b, PILOT_EXTRA)}/result.json"):
                rows.append((lr_b, PILOT_EXTRA))
        results = {}
        for lr, b in rows:
            p = f"{pilot_dir(rung, lr, b)}/result.json"
            if not os.path.exists(p):
                lines.append(f"| {lr:g} | {b:,} | (not run) | | |")
                continue
            r = json.load(open(p))
            results[(lr, b)] = r["dev_full_loss"]
            lines.append(f"| {lr:g} | {b:,} | {r['dev_full_loss']:.4f} | {r['steps']:,} | {r['elapsed']:.0f} |")
        lines.append("")
        if lr_b is not None and all((lr_b, b) in results for b in PILOT_BUDGETS):
            budgets = [b for lr, b in rows if lr == lr_b and (lr, b) in results]
            best = min(results[(lr_b, b)] for b in budgets)
            pick = min(b for b in budgets if results[(lr_b, b)] <= best * (1 + WITHIN))
            lines += [f"**Chosen for {rung}:** peak LR {lr_b:g}, budget {pick:,} symbols (smallest within "
                      f"{WITHIN:.0%} of the best, {best:.4f}).", ""]
    with open("logs/ladder/pilot.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "pilot_lr", "pilot_budget", "pilot_report", "rung"])
    ap.add_argument("--rung", default="L1")
    ap.add_argument("--arm", default="ln", choices=list(ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "counts":
        counts()
    elif a.cmd == "pilot_lr":
        for lr in PILOT_LRS:
            run(config(a.rung, "ln", 0, lr, PILOT_BUDGETS[0], pilot_dir(a.rung, lr, PILOT_BUDGETS[0]), a.device))
        while best_lr(a.rung) == lrs_run(a.rung)[-1]:
            lr = round(lrs_run(a.rung)[-1] * PILOT_LR_STEP, 6)
            run(config(a.rung, "ln", 0, lr, PILOT_BUDGETS[0], pilot_dir(a.rung, lr, PILOT_BUDGETS[0]), a.device))
    elif a.cmd == "pilot_budget":
        lr = best_lr(a.rung)
        res = {b: run(config(a.rung, "ln", 0, lr, b, pilot_dir(a.rung, lr, b), a.device))["dev_full_loss"]
               for b in PILOT_BUDGETS}
        if res[PILOT_BUDGETS[-1]] < res[PILOT_BUDGETS[-2]] * (1 - WITHIN):
            run(config(a.rung, "ln", 0, lr, PILOT_EXTRA, pilot_dir(a.rung, lr, PILOT_EXTRA), a.device))
    elif a.cmd == "pilot_report":
        pilot_report()
    elif a.cmd == "rung":
        assert a.arm == "ln" or a.rung in LN_ARM_RUNGS, f"arm {a.arm} is trained only at {LN_ARM_RUNGS}"
        assert a.rung in CHOSEN, f"no pilot choice for {a.rung} yet (CHOSEN in this file, from logs/ladder/pilot.md)"
        lr, budget = CHOSEN[a.rung]
        for s in [int(x) for x in a.seeds.split(",")]:
            run(config(a.rung, a.arm, s, lr, budget, f"{ROOT}/{a.rung}_{a.arm}_s{s}", a.device))


if __name__ == "__main__":
    main()
