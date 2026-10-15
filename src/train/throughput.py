"""Throughput test: run training steps on random tokens for a fixed wall time.

Usage: python -m src.train.throughput --device cpu --seconds 120 --variant mlp
Prints tokens/sec. Used to extrapolate pilot and final-run costs (AGENTS.md section 5).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from src.train.model import ModelConfig, TinyLM, param_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--variant", default="mlp", choices=["attn", "mlp"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (GPU only)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n_blocks", type=int, default=1)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    dev = torch.device(args.device)
    cfg = ModelConfig(mlp=args.variant == "mlp", n_blocks=args.n_blocks)
    model = TinyLM(cfg).to(dev)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-3, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=args.amp)

    def step():
        x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.context_length + 1), device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=args.amp):
            logits = model(x[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1)
            )
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(5):  # warmup
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        loss = step()
        n += 1
        if n % 10 == 0:
            loss.item()
    loss.item()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tps = n * args.batch * cfg.context_length / dt
    print(json.dumps({
        "device": args.device, "variant": args.variant, "n_blocks": args.n_blocks, "amp": args.amp,
        "compile": args.compile,
        "threads": torch.get_num_threads(), "steps": n, "seconds": round(dt, 1),
        "tokens_per_sec": round(tps), "torch": torch.__version__,
    }))


if __name__ == "__main__":
    main()
