"""Drivers for the pilot (AGENTS.md section 5) and the 10 final runs. Every run is resumable and
skipped when its result.json exists, so a disconnected session just re-runs the same command.

  python -m src.train.runs pilot_lr                 # 3 runs, mlp seed 0, smallest budget
  python -m src.train.runs pilot_budget             # budget ladder at the best LR
  python -m src.train.runs pilot_report             # writes logs/pilot.md
  python -m src.train.runs final --budget_tokens N  # 10 models -> models/final/{variant}_s{seed}
  python -m src.train.runs deep_lr                  # side study: 4-block attn, seed 0, 50M tokens, LR 1e-3 vs 2e-3
  python -m src.train.runs deep --n_blocks 2 --seeds 0,1,2,3,4 [--peak_lr 2e-3]
                                                    # side study: models/deep/attn{L}L_s{seed}, one epoch each
  python -m src.train.runs readable_dev [--arms none,O,I,control] [--lams 0.1,0.3,1,3,10] [--extra I_att,I_vote --lam X]
                                                    # side study "trained to be readable": seed 0, 100M tokens
  python -m src.train.runs readable --arm O|I --lam X --seeds 1,2,3,4
                                                    # side study: models/readable/{arm}_l{lam}_s{seed}, one epoch each
  python -m src.train.runs fixed_lr                 # study A (fixed votes, rule R1): seed 0, 50M tokens, three LRs
  python -m src.train.runs fixed --name base --seeds 0,1,2,3,4 --peak_lr X
                                                    # study A: models/fixed_votes/{name}_s{seed}, one epoch each;
                                                    # names: base (next,copy,topic,skip2), shift (head 3 votes skip3)
  python -m src.train.runs law_dev [--arms both] [--lams 1,3,10,30,100]
                                                    # study B (penalty against the law): seed 0, 100M tokens
  python -m src.train.runs law --arm both|attn|vote --lam X --seeds 1,2,3,4
                                                    # study B: models/law_pressure/{arm}_l{lam}_s{seed}, one epoch each
  python -m src.train.runs law_restore --from models/law_pressure/both_l10_s1 [--seeds ...]
                                                    # study B: continue without the penalty (restoring force)
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
FROZEN_PEAK_LR = 2e-3                          # configs/frozen.yaml (chosen by the pilot)
DEEP_LR_TOKENS = 50_000_000                    # side-study LR check
READABLE_DEV_TOKENS = 100_000_000              # side study "trained to be readable": development grid
FIXED_LR_TOKENS = 50_000_000                   # study A, rule R1
FIXED_LRS = [1e-3, 2e-3, 4e-3]
FIXED_TABLES = {"base": "next,copy,topic,skip2", "shift": "next,copy,topic,skip3"}   # one vote table per head
LAW_DEV_TOKENS = 100_000_000                   # study B: development grid
RESTORE_STEPS, RESTORE_LR, RESTORE_SEED_OFFSET = 3000, 2e-4, 10   # study B: continuation without the penalty


def readable_kw(arm: str, lam: float, c: float) -> dict:
    """TrainConfig settings for one readable-training run (configs/side_readable.yaml; c from rule R2)."""
    if arm == "none":                                  # control: aux metrics logged, no pressure
        return {"readable_arm": "O", "lambda_out": 0.0}
    if arm == "O":
        return {"readable_arm": "O", "lambda_out": lam}
    if arm == "I":
        return {"readable_arm": "I", "lambda_att": lam, "lambda_vote": lam * c}
    if arm == "I_att":
        return {"readable_arm": "I", "lambda_att": lam, "lambda_vote": 0.0}
    if arm == "I_vote":
        return {"readable_arm": "I", "lambda_att": 0.0, "lambda_vote": lam * c}
    raise ValueError(arm)


def readable_rules() -> dict:
    p = os.path.join("analysis", "out", "readable_training", "rules.json")
    return json.load(open(p)) if os.path.exists(p) else {}


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
    ap.add_argument("cmd", choices=["pilot_lr", "pilot_budget", "pilot_report", "final", "deep_lr", "deep",
                                    "readable_dev", "readable", "fixed_lr", "fixed", "law_dev", "law",
                                    "law_restore"])
    ap.add_argument("--root", default="models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--budget_tokens", type=int, default=None)
    ap.add_argument("--peak_lr", type=float, default=None)
    ap.add_argument("--only", default=None, help="comma list like attn_s0,mlp_s3 (final only)")
    ap.add_argument("--n_blocks", type=int, default=2, help="deep only")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="deep and readable")
    ap.add_argument("--arms", default="none,O,I,control", help="readable_dev")
    ap.add_argument("--lams", default="0.1,0.3,1,3,10", help="readable_dev")
    ap.add_argument("--extra", default="", help="readable_dev: comma list of I_att, I_vote at --lam")
    ap.add_argument("--arm", default="O", help="readable")
    ap.add_argument("--lam", type=float, default=None, help="readable, readable_dev --extra")
    ap.add_argument("--name", default="base", help="fixed: " + ", ".join(FIXED_TABLES))
    ap.add_argument("--from", dest="from_dir", default="", help="law_restore: run folder whose final.pt to continue")
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
    elif args.cmd == "deep_lr":
        # side study (logs/decisions.md item 8): LR check for the deepest model, attention-only, seed 0
        for lr in (1e-3, 2e-3):
            _run(args.root, f"deep/pilot/attn4L_s0_50M_lr{lr:g}", args.device, args.amp, variant="attn", seed=0,
                 n_blocks=4, budget_tokens=DEEP_LR_TOKENS, peak_lr=lr)
    elif args.cmd in ("readable_dev", "readable"):
        run_readable(args)
    elif args.cmd == "fixed_lr":
        for lr in FIXED_LRS:
            _run(args.root, f"fixed_votes/lr/base_s0_50M_lr{lr:g}", args.device, args.amp, variant="fixed", seed=0,
                 fixed_tables=FIXED_TABLES["base"], budget_tokens=FIXED_LR_TOKENS, peak_lr=lr)
    elif args.cmd == "law_dev":
        arms = args.arms.split(",") if args.arms != "none,O,I,control" else ["both"]
        for arm in arms:
            for lam in (float(x) for x in args.lams.split(",")):
                _run(args.root, f"law_pressure/dev/{arm}_l{lam:g}_s0_100M", args.device, args.amp, variant="attn",
                     seed=0, budget_tokens=args.budget_tokens or LAW_DEV_TOKENS, peak_lr=FROZEN_PEAK_LR, law_arm=arm,
                     lambda_law=lam)
    elif args.cmd == "law":
        assert args.lam is not None, "--lam"
        for seed in (int(s) for s in args.seeds.split(",")):
            _run(args.root, f"law_pressure/{args.arm}_l{args.lam:g}_s{seed}", args.device, args.amp, variant="attn",
                 seed=seed, budget_tokens=args.budget_tokens or LADDER[-1], peak_lr=FROZEN_PEAK_LR, law_arm=args.arm,
                 lambda_law=args.lam)
    elif args.cmd == "law_restore":
        # continue a model without the penalty: fresh optimizer, 1% warmup and cosine at RESTORE_LR, another batch order
        name = os.path.basename(os.path.normpath(args.from_dir))
        seed = int(name.rsplit("_s", 1)[1])
        _run(args.root, f"law_pressure/restore/{name}", args.device, args.amp, variant="attn",
             seed=seed + RESTORE_SEED_OFFSET, init_from=os.path.join(args.from_dir, "final.pt"),
             budget_tokens=RESTORE_STEPS * 64 * 256, peak_lr=RESTORE_LR)
    elif args.cmd == "fixed":
        assert args.peak_lr, "fixed needs --peak_lr (rule R1, analysis/out/fixed_votes/rules.json)"
        for seed in (int(s) for s in args.seeds.split(",")):
            _run(args.root, f"fixed_votes/{args.name}_s{seed}", args.device, args.amp, variant="fixed", seed=seed,
                 fixed_tables=FIXED_TABLES[args.name], budget_tokens=args.budget_tokens or LADDER[-1],
                 peak_lr=args.peak_lr)
    elif args.cmd == "deep":
        # side study: one epoch per model with the frozen hyperparameters; seed s has the batch order of attn_s{s}
        for seed in (int(s) for s in args.seeds.split(",")):
            _run(args.root, f"deep/attn{args.n_blocks}L_s{seed}", args.device, args.amp, variant="attn", seed=seed,
                 n_blocks=args.n_blocks, budget_tokens=args.budget_tokens or LADDER[-1],
                 peak_lr=args.peak_lr or FROZEN_PEAK_LR)


def run_readable(args) -> None:
    rules = readable_rules()
    c = rules.get("R2", {}).get("c")
    aux_windows = rules.get("R6", {}).get("aux_windows", 16)
    common = {"variant": "attn", "peak_lr": args.peak_lr or FROZEN_PEAK_LR, "aux_windows": aux_windows}
    if args.cmd == "readable_dev":
        runs = []
        lams = [float(x) for x in args.lams.split(",")]
        for arm in args.arms.split(","):
            if arm == "none":
                runs.append(("readable/dev/none_s0_100M", "none", 0.0, {}))
            elif arm == "control":                     # positive Goodhart control: arm O, lambda 10, 256-window pool
                runs.append(("readable/dev/O_l10_pool256_s0_100M", "O", 10.0, {"readable_pool_limit": 256}))
            elif arm in ("O", "I"):
                runs += [(f"readable/dev/{arm}_l{lam:g}_s0_100M", arm, lam, {}) for lam in lams]
        for arm in (x for x in args.extra.split(",") if x):
            runs.append((f"readable/dev/{arm}_l{args.lam:g}_s0_100M", arm, args.lam, {}))
        for name, arm, lam, kw in runs:
            if arm.startswith("I"):
                assert c is not None, "rule R2 (c) must be applied before arm I runs"
            ramp = rules.get("R5", {}).get(f"ramp_{'I' if arm.startswith('I') else 'O'}", 0.0)
            _run(args.root, name, args.device, args.amp, seed=0, budget_tokens=args.budget_tokens or READABLE_DEV_TOKENS,
                 readable_ramp=ramp, **common, **readable_kw(arm, lam, c or 0.0), **kw)
    else:
        assert args.lam is not None, "--lam"
        if args.arm.startswith("I"):
            assert c is not None, "rule R2 (c) must be applied before arm I runs"
        ramp = rules.get("R5", {}).get(f"ramp_{args.arm}", 0.0)
        for seed in (int(s) for s in args.seeds.split(",")):
            _run(args.root, f"readable/{args.arm}_l{args.lam:g}_s{seed}", args.device, args.amp, seed=seed,
                 budget_tokens=args.budget_tokens or LADDER[-1], readable_ramp=ramp, **common,
                 **readable_kw(args.arm, args.lam, c or 0.0))


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
