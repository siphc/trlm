"""Shared helpers for the decision-rule scripts of studies A and B (analysis/fixed_votes/rules.py,
analysis/law_pressure/rules.py): a merged rules.json per study, and step timing on the current device."""
from __future__ import annotations

import json
import os
import shutil
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_rules(study: str, update: dict) -> dict:
    d = os.path.join(REPO, "analysis", "out", study)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "rules.json")
    rules = json.load(open(p)) if os.path.exists(p) else {}
    rules.update(update)
    with open(p, "w") as f:
        json.dump(rules, f, indent=1)
    return rules


def load_rules(study: str) -> dict:
    p = os.path.join(REPO, "analysis", "out", study, "rules.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def seconds_per_step(device: str, steps: int = 150, **kw) -> float:
    """Mean seconds per training step between steps 50 and `steps` of a throwaway run (batch 64 x 256)."""
    from src.train.train import TrainConfig, train

    tmp = tempfile.mkdtemp(prefix="rule_timing_")
    try:
        train(TrainConfig(seed=0, budget_tokens=steps * 64 * 256, peak_lr=2e-3, device=device, eval_every=10**9,
                          log_every=50, out=tmp, **kw))
        rows = {r["step"]: r["elapsed"] for r in map(json.loads, open(os.path.join(tmp, "log.jsonl")))}
        return (rows[steps] - rows[50]) / (steps - 50)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def device_name(device: str) -> str:
    import platform

    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"{device} ({platform.machine()}, {platform.platform()})"
