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
import torch.nn.functional as F

from src.train.data import BatchSampler, load_eval, load_stream
from src.train.model import ModelConfig, TinyLM, param_groups


@dataclass
class TrainConfig:
    variant: str = "mlp"  # "attn" or "mlp"
    seed: int = 0
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

    @property
    def tokens_per_step(self) -> int:
        return self.batch_windows * self.context_length

    @property
    def total_steps(self) -> int:
        return max(1, self.budget_tokens // self.tokens_per_step)


def model_config(variant: str) -> ModelConfig:
    assert variant in ("attn", "mlp")
    return ModelConfig(mlp=(variant == "mlp"))


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
    model = TinyLM(model_config(cfg.variant)).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.peak_lr, betas=cfg.betas)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler = BatchSampler(load_stream("train"), cfg.seed, cfg.batch_windows)
    dev_fast = load_eval("dev_fast")

    step = 0
    latest = os.path.join(cfg.out, "latest.pt")
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
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % cfg.log_every == 0 or step == cfg.total_steps:
            row = {"step": step, "tokens": step * cfg.tokens_per_step, "lr": lr,
                   "train_loss": loss.item(), "grad_norm": float(gnorm), "elapsed": time.time() - t0}
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                row["dev_fast_loss"] = evaluate(model, dev_fast, device)
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
    save_weights(model, os.path.join(cfg.out, "final.pt"),
                 {"step": step, "tokens": step * cfg.tokens_per_step, "dev_full_loss": dev_full})
    result = {"variant": cfg.variant, "seed": cfg.seed, "budget_tokens": cfg.budget_tokens,
              "peak_lr": cfg.peak_lr, "steps": step, "tokens": step * cfg.tokens_per_step,
              "dev_full_loss": dev_full, "elapsed": time.time() - t0, "device": str(device)}
    with open(os.path.join(cfg.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    logf.close()
    print(json.dumps(result))
    return result


def load_model(path: str, device: str = "cpu") -> TinyLM:
    """Load an fp32 model from a weights checkpoint (final.pt / ckpt_step*.pt)."""
    st = torch.load(path, map_location=device, weights_only=False)
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
