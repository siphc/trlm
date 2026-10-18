"""Shared pieces of the side study "can a language model be trained to be readable?" (logs/readable_training.md).

Settings come from configs/side_readable.yaml. The target pool lives in data/readable/pool/ (not in git).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.train.data import load_stream, n_windows, window  # noqa: E402

CONFIG = os.path.join(REPO, "configs", "side_readable.yaml")
OUT = os.path.join(REPO, "analysis", "out", "readable_training")
RULES = os.path.join(OUT, "rules.json")          # values fixed by the pre-registered rules R1-R7
RULE_HEADS = ("head0", "head1", "head2")         # program rules LOCAL-0, RECALL, LOCAL-2 -> model heads 0, 1, 2
T = 256


def config() -> dict:
    import yaml

    return yaml.safe_load(open(CONFIG))


def pool_dir() -> str:
    return os.path.join(REPO, config()["pool"]["path"])


def train_window_index(n: int, seed: int) -> np.ndarray:
    """Window indices of analysis.fitlib.train_windows(n, seed) (same draw, unsorted order irrelevant)."""
    total = n_windows(len(load_stream("train")))
    return np.random.default_rng(seed).choice(total, size=n, replace=False)


def excluded_windows() -> set:
    out = set()
    for n, seed in config()["pool"]["exclude"]:
        out.update(int(i) for i in train_window_index(n, seed))
    return out


def pool_indices() -> tuple[np.ndarray, np.ndarray]:
    """(train, heldout) TRAIN-window indices of the pool: the first windows of default_rng(rng).permutation that are
    not excluded."""
    cfg = config()["pool"]
    total = n_windows(len(load_stream("train")))
    excl = excluded_windows()
    perm = np.random.default_rng(cfg["rng"]).permutation(total)
    keep = []
    need = cfg["n_train"] + cfg["n_heldout"]
    for i in perm:
        if int(i) not in excl:
            keep.append(int(i))
            if len(keep) == need:
                break
    keep = np.asarray(keep, dtype=np.int64)
    return keep[: cfg["n_train"]], keep[cfg["n_train"]:]


def windows_at(idx: np.ndarray) -> np.ndarray:
    s = load_stream("train")
    return np.stack([np.asarray(window(s, int(i))) for i in idx]).astype(np.int64)


def load_rules() -> dict:
    return json.load(open(RULES)) if os.path.exists(RULES) else {}


def save_rules(update: dict) -> dict:
    os.makedirs(OUT, exist_ok=True)
    r = load_rules()
    r.update(update)
    with open(RULES, "w") as f:
        json.dump(r, f, indent=1)
    return r
