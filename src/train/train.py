"""Training loop. Plain PyTorch, resumable from any checkpoint.

Usage:
  python -m src.train.train --variant mlp --seed 0 --budget_tokens 50_000_000 --out models/pilot/mlp_s0_50M
  (re-running the same command resumes from <out>/latest.pt if it exists)

Checkpoints in <out>/:
  ckpt_step{N}.pt   ~20 log-spaced in tokens seen (model weights only, fp32)
  final.pt          final fp32 weights
  latest.pt         full resume state (model, optimizer, scaler, step), refreshed every few minutes
  log.jsonl         one JSON row per logged step
  config.json       the resolved TrainConfig
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.data import BatchSampler, load_eval, load_stream
from src.train.model import ModelConfig, TinyLM, param_groups


@dataclass
class TrainConfig:
    variant: str = "mlp"  # "attn" or "mlp"
    seed: int = 0
    n_blocks: int = 1     # the main track is 1 block (configs/frozen.yaml); > 1 only for side studies
    budget_tokens: int = 12_500_000
    peak_lr: float = 1e-3
    betas: tuple = (0.9, 0.95)
    warmup_frac: float = 0.01
    final_lr_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_windows: int = 64
    context_length: int = 256
    n_log_checkpoints: int = 20
    out: str = "models/scratch"
    device: str = "auto"
    amp: bool = False  # fp16 autocast, CUDA only
    log_every: int = 50
    eval_every: int = 500
    resume_every_sec: float = 300.0
    # side study "can a language model be trained to be readable?" (logs/decisions.md item 9; src/train/readable.py).
    # readable_arm "" leaves training unchanged.
    readable_arm: str = ""            # "", "O" (output level) or "I" (internal level)
    readable_pool: str = "data/readable/pool"
    readable_pool_limit: int = 0      # use only the first N pool windows (positive control); 0 = all
    readable_targets: str = "data/readable/vote_targets.npz"
    lambda_out: float = 0.0
    lambda_att: float = 0.0
    lambda_vote: float = 0.0
    aux_windows: int = 16             # auxiliary pool windows per step
    readable_ramp: float = 0.0        # fraction of the steps over which the lambdas ramp up linearly (rule R5)
    # study A "fixed_votes" (logs/decisions.md item 10; src/train/fixed_votes.py): heads vote with fixed corpus tables,
    # one table name per head. "" trains the usual TinyLM.
    fixed_tables: str = ""
    fixed_tables_path: str = "data/fixed_votes/tables.npz"
    # study B "law_pressure" (logs/decisions.md item 10; src/train/law_pressure.py): a penalty against the
    # look-follows-write law. law_arm "" leaves training unchanged.
    law_arm: str = ""                 # "", "both", "attn" or "vote"
    lambda_law: float = 0.0
    law_ramp: float = 0.05            # fraction of the steps over which lambda_law ramps up linearly
    law_windows: int = 0              # compute the penalty on the first N windows of each batch; 0 = all (rule R2)
    law_p1_path: str = "data/fixed_votes/tables.npz"   # FOLLOW's bigram rule ('follow')
    init_from: str = ""               # start from these weights (fresh optimizer and schedule); "" = random init

    @property
    def tokens_per_step(self) -> int:
        return self.batch_windows * self.context_length

    @property
    def total_steps(self) -> int:
        return max(1, self.budget_tokens // self.tokens_per_step)


def model_config(variant: str, n_blocks: int = 1) -> ModelConfig:
    assert variant in ("attn", "mlp")
    return ModelConfig(mlp=(variant == "mlp"), n_blocks=n_blocks)


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.fixed_tables:
        from src.train.fixed_votes import FixedVoteConfig, FixedVoteLM

        assert not cfg.readable_arm, "fixed-vote models take no readable-training loss"
        return FixedVoteLM(FixedVoteConfig(tables=cfg.fixed_tables), cfg.fixed_tables_path)
    return TinyLM(model_config(cfg.variant, cfg.n_blocks))


def lr_at(step: int, cfg: TrainConfig) -> float:
    """1% linear warmup to peak, then cosine decay to final_lr_frac * peak at the last step."""
    total = cfg.total_steps
    warm = max(1, int(round(cfg.warmup_frac * total)))
    if step < warm:
        return cfg.peak_lr * (step + 1) / warm
    progress = (step - warm) / max(1, total - 1 - warm)
    progress = min(1.0, progress)
    lo = cfg.final_lr_frac
    return cfg.peak_lr * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * progress)))


def checkpoint_steps(cfg: TrainConfig) -> list[int]:
    """~n_log_checkpoints steps log-spaced in tokens seen, from 16 steps up to (excluding) the end."""
    total = cfg.total_steps
    pts = np.geomspace(min(16, total), total, cfg.n_log_checkpoints + 1)[:-1]
    return sorted(set(int(round(p)) for p in pts if 0 < round(p) < total))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: TinyLM, wins: np.ndarray, device: torch.device, bs: int = 50) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(wins), bs):
        x = torch.as_tensor(wins[i : i + bs].astype(np.int64), device=device)
        logits = model(x[:, :-1]).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        tot += loss.item()
        n += x[:, 1:].numel()
    model.train()
    return tot / n


def save_weights(model: TinyLM, path: str, extra: dict) -> None:
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    tmp = path + ".tmp"
    torch.save({"model": sd, "model_config": dataclasses.asdict(model.cfg), **extra}, tmp)
    os.replace(tmp, path)


def train(cfg: TrainConfig) -> dict:
    os.makedirs(cfg.out, exist_ok=True)
    device = pick_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump({**dataclasses.asdict(cfg), "resolved_device": str(device), "amp_active": use_amp,
                   "torch": torch.__version__, "total_steps": cfg.total_steps}, f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.peak_lr, betas=cfg.betas)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler = BatchSampler(load_stream("train"), cfg.seed, cfg.batch_windows)
    dev_fast = load_eval("dev_fast")
    aux = None
    if cfg.readable_arm:
        from src.train.readable import ReadableAux

        aux = ReadableAux(cfg, device)
        aux.check_or_write(cfg.out)
    law = None
    if cfg.law_arm:
        from src.train.law_pressure import LawPenalty

        law = LawPenalty(cfg, device)

    step = 0
    latest = os.path.join(cfg.out, "latest.pt")
    if cfg.init_from and not os.path.exists(latest):
        model.load_state_dict(torch.load(cfg.init_from, map_location=device, weights_only=False)["model"])
    if os.path.exists(latest):
        st = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        torch.set_rng_state(st["torch_rng"].cpu())  # map_location may have moved it to the GPU
        print(f"resumed from step {step}")

    ckpts = set(checkpoint_steps(cfg))
    log_path = os.path.join(cfg.out, "log.jsonl")
    if step == 0 and os.path.exists(log_path):
        os.remove(log_path)
    logf = open(log_path, "a")
    t0 = time.time()
    last_resume_save = time.time()
    model.train()

    def save_resume():
        tmp = latest + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, latest)

    while step < cfg.total_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.as_tensor(sampler.batch(step), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        total = loss
        if aux is not None:
            aux_loss, aux_info = aux.loss(model, step)
            total = loss + aux_loss
        if law is not None:
            law_loss, law_info = law.loss(model, x, step)
            total = total + law_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % cfg.log_every == 0 or step == cfg.total_steps:
            row = {"step": step, "tokens": step * cfg.tokens_per_step, "lr": lr,
                   "train_loss": loss.item(), "grad_norm": float(gnorm), "elapsed": time.time() - t0}
            if aux is not None:
                row.update({k: float(v) for k, v in aux_info.items()})
            if law is not None:
                row.update({k: float(v) for k, v in law_info.items()})
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                row["dev_fast_loss"] = evaluate(model, dev_fast, device)
                if aux is not None:
                    row.update(aux.metrics(model))
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            if "dev_fast_loss" in row:
                print(f"step {step}/{cfg.total_steps} tok {row['tokens']:,} lr {lr:.2e} "
                      f"train {row['train_loss']:.4f} dev_fast {row['dev_fast_loss']:.4f} "
                      f"({row['elapsed']:.0f}s)", flush=True)

        if step in ckpts:
            save_weights(model, os.path.join(cfg.out, f"ckpt_step{step}.pt"),
                         {"step": step, "tokens": step * cfg.tokens_per_step})
        if time.time() - last_resume_save > cfg.resume_every_sec:
            save_resume()
            last_resume_save = time.time()

    save_resume()
    dev_full = evaluate(model, load_eval("dev_full"), device)
    extra = {"readable": aux.summary()} if aux is not None else {}
    if law is not None:
        extra["law_pressure"] = law.summary()
    save_weights(model, os.path.join(cfg.out, "final.pt"),
                 {"step": step, "tokens": step * cfg.tokens_per_step, "dev_full_loss": dev_full, **extra})
    result = {"variant": cfg.variant, "seed": cfg.seed, "n_blocks": cfg.n_blocks, "budget_tokens": cfg.budget_tokens,
              "peak_lr": cfg.peak_lr, "steps": step, "tokens": step * cfg.tokens_per_step,
              "dev_full_loss": dev_full, "elapsed": time.time() - t0, "device": str(device), **extra}
    with open(os.path.join(cfg.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    logf.close()
    print(json.dumps(result))
    return result


def load_model(path: str, device: str = "cpu") -> TinyLM:
    """Load an fp32 model from a weights checkpoint (final.pt / ckpt_step*.pt)."""
    st = torch.load(path, map_location=device, weights_only=False)
    if st["model_config"].get("arch") == "fixed_votes":
        from src.train.fixed_votes import load

        return load(st, device)
    model = TinyLM(ModelConfig(**st["model_config"]))
    model.load_state_dict(st["model"])
    return model.float().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        if f.name == "betas":
            continue
        typ = {bool: lambda s: s.lower() in ("1", "true", "yes")}.get(type(f.default), type(f.default))
        ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
