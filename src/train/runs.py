"""Drivers for the pilot (AGENTS.md section 5) and the 10 final runs. Every run is resumable and
skipped when its result.json exists, so a disconnected session just re-runs the same command.

  python -m src.train.runs pilot_lr                 # 3 runs, mlp seed 0, smallest budget
  python -m src.train.runs pilot_budget             # budget ladder at the best LR
  python -m src.train.runs pilot_report             # writes logs/pilot.md
  python -m src.train.runs final --budget_tokens N  # 10 models -> models/final/{variant}_s{seed}
  Add --root /content/drive/MyDrive/trlm on Colab to write checkpoints to Drive.
"""
from __future__ import annotations

import argparse
import json
import os

from src.train.train import TrainConfig, train

LRS = [5e-4, 1e-3, 2e-3]
EPOCH_TOKENS = 527_489_811                     # data/prep_meta.json n_tokens.train
EPOCH_STEPS = (EPOCH_TOKENS - 1) // 256 // 64  # whole batches in one pass over the train windows
LADDER = [12_500_000, 50_000_000, 200_000_000, EPOCH_STEPS * 64 * 256]
WITHIN = 0.01                                  # choose smallest budget within 1% of the best run


def _run(root: str, name: str, device: str, amp: bool, **kw) -> dict:
    out = os.path.join(root, name)
    res_path = os.path.join(out, "result.json")
    if os.path.exists(res_path):
        return json.load(open(res_path))
    return train(TrainConfig(out=out, device=device, amp=amp, **kw))


def lr_name(lr: float) -> str:
    return f"pilot/lr{lr:g}_mlp_s0_{LADDER[0] // 1_000_000}M".replace(".", "p")


def best_lr(root: str) -> float:
    res = {lr: json.load(open(os.path.join(root, lr_name(lr), "result.json"))) for lr in LRS}
    return min(res, key=lambda lr: res[lr]["dev_full_loss"])


def budget_name(b: int, lr: float) -> str:
    return f"pilot/budget{b // 1_000_000}M_mlp_s0_lr{lr:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot_lr", "pilot_budget", "pilot_report", "final"])
    ap.add_argument("--root", default="models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--budget_tokens", type=int, default=None)
    ap.add_argument("--peak_lr", type=float, default=None)
    ap.add_argument("--only", default=None, help="comma list like attn_s0,mlp_s3 (final only)")
    args = ap.parse_args()

    if args.cmd == "pilot_lr":
        for lr in LRS:
            _run(args.root, lr_name(lr), args.device, args.amp, variant="mlp", seed=0,
                 budget_tokens=LADDER[0], peak_lr=lr)
    elif args.cmd == "pilot_budget":
        lr = best_lr(args.root)
        for b in LADDER:
            name = lr_name(lr) if b == LADDER[0] else budget_name(b, lr)
            _run(args.root, name, args.device, args.amp, variant="mlp", seed=0, budget_tokens=b,
                 peak_lr=lr)
    elif args.cmd == "pilot_report":
        report(args.root)
    elif args.cmd == "final":
        assert args.budget_tokens and args.peak_lr, "final needs --budget_tokens and --peak_lr from logs/pilot.md"
        only = set(args.only.split(",")) if args.only else None
        for variant in ("attn", "mlp"):
            for seed in range(5):
                mid = f"{variant}_s{seed}"
                if only and mid not in only:
                    continue
                _run(args.root, f"final/{mid}", args.device, args.amp, variant=variant, seed=seed,
                     budget_tokens=args.budget_tokens, peak_lr=args.peak_lr)


def report(root: str) -> None:
    rows = []
    for lr in LRS:
        r = json.load(open(os.path.join(root, lr_name(lr), "result.json")))
        rows.append(("lr", lr, r))
    lr = best_lr(root)
    budget_rows = []
    for b in LADDER:
        name = lr_name(lr) if b == LADDER[0] else budget_name(b, lr)
        p = os.path.join(root, name, "result.json")
        if os.path.exists(p):
            budget_rows.append((b, json.load(open(p))))
    best = min(r["dev_full_loss"] for _, r in budget_rows)
    chosen = next(b for b, r in budget_rows if r["dev_full_loss"] <= best * (1 + WITHIN))
    lines = ["# Pilot", "", "## LR check (mlp seed 0, %sM tokens)" % (LADDER[0] // 1_000_000), "",
             "| peak LR | dev_full loss | tokens | seconds | device |", "|---|---|---|---|---|"]
    for _, l, r in rows:
        lines.append(f"| {l:g} | {r['dev_full_loss']:.4f} | {r['tokens']:,} | {r['elapsed']:.0f} | {r['device']} |")
    lines += ["", f"Best LR: {lr:g}", "", "## Budget ladder (mlp seed 0, best LR, full cosine each)", "",
              "| budget tokens | dev_full loss | rel. to best | seconds | device |", "|---|---|---|---|---|"]
    for b, r in budget_rows:
        lines.append(f"| {r['tokens']:,} | {r['dev_full_loss']:.4f} | {r['dev_full_loss'] / best - 1:+.2%} | "
                     f"{r['elapsed']:.0f} | {r['device']} |")
    lines += ["", f"Chosen budget (smallest within {WITHIN:.0%} of best): **{chosen:,} tokens** at peak LR {lr:g}", ""]
    os.makedirs("logs", exist_ok=True)
    with open("logs/pilot.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
