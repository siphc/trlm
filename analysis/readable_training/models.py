"""Model handling for the readable-training measurements: which head plays which program role, and a head-aligned
copy of a checkpoint in which the roles LOCAL-0, RECALL, LOCAL-2 sit on heads 0, 1, 2 (heads are interchangeable, so
the permutation is an exact symmetry), so that every tool that names heads 0-2 (claim tests, tables_vs_attention,
mech_eval) works unchanged.

  .venv/bin/python -m analysis.readable_training.models align <model id or final.pt> [...]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch

from analysis.readable_training.common import OUT, REPO, RULE_HEADS

PATTERN_WINDOWS = (16, 9125)            # configs/side_readable.yaml eval_sets.pattern


def name_of(model) -> str:
    """'attn_s1' -> 'attn_s1'; 'models/readable/O_l1_s2/final.pt' -> 'readable_O_l1_s2'."""
    p = str(model)
    if not p.endswith(".pt"):
        return p
    d = os.path.basename(os.path.dirname(os.path.abspath(p)))
    parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
    return f"{parent}_{d}" if parent in ("readable", "dev") else d


def model_path(model) -> str:
    from harness.common import model_path as mp

    return mp(str(model))


def role_assignment(model) -> dict:
    """Pattern KL of each v1.3 rule (scalars unchanged) against each model head on train_windows(16, 9125); the
    assignment of the three rules to distinct heads with the lowest total KL."""
    import importlib

    from analysis.attention_fit import model_attention, pattern_ce
    from analysis.fitlib import train_windows
    from analysis.mech_eval import stats

    s = stats()
    W = train_windows(*PATTERN_WINDOWS)
    from harness.common import load_model

    A = model_attention(load_model(model_path(model)), W)
    table = {}
    for r in RULE_HEADS:
        mod = importlib.import_module(f"program.{r}")
        for h in range(A.shape[1]):
            table[(r, h)] = float(pattern_ce(mod, A[:, h], W, s)[1])
    best = min(itertools.permutations(range(A.shape[1]), 3), key=lambda hs: sum(table[(r, h)] for r, h in zip(RULE_HEADS, hs)))
    return {"assignment": dict(zip(RULE_HEADS, map(int, best))),
            "pattern_kl": {f"{r}->h{h}": v for (r, h), v in table.items()}}


def permute_heads(state: dict, order: list[int], n_heads=4, d_model=128) -> dict:
    """New head i = old head order[i]: rows of the q, k and v blocks of attn.qkv.weight, columns of attn.out.weight."""
    dh = d_model // n_heads
    sd = {k: v.clone() for k, v in state.items()}
    qkv = state["blocks.0.attn.qkv.weight"]
    out = state["blocks.0.attn.out.weight"]
    rows = [blk * d_model + old * dh + j for blk in range(3) for old in order for j in range(dh)]
    sd["blocks.0.attn.qkv.weight"] = qkv[rows]
    cols = [old * dh + j for old in order for j in range(dh)]
    sd["blocks.0.attn.out.weight"] = out[:, cols]
    return sd


def align(model) -> dict:
    """Write analysis/out/readable_training/<name>/aligned.pt (roles on heads 0, 1, 2; the other head last) and
    roles.json; check that the logits are unchanged."""
    from src.train.data import load_eval
    from src.train.train import load_model

    name = name_of(model)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    roles = role_assignment(model)
    a = roles["assignment"]
    order = [a["head0"], a["head1"], a["head2"]]
    order += [h for h in range(4) if h not in order]
    src = torch.load(model_path(model), map_location="cpu", weights_only=False)
    out = dict(src)
    out["model"] = permute_heads(src["model"], order)
    out["aligned_from"] = {"path": model_path(model), "order": order}
    dst = os.path.join(d, "aligned.pt")
    torch.save(out, dst)
    x = torch.as_tensor(load_eval("dev_fast")[:4, :256].astype(np.int64))
    with torch.no_grad():
        diff = float((load_model(model_path(model))(x) - load_model(dst)(x)).abs().max())
    assert diff < 1e-4, diff
    roles.update({"order_new_head_from_old": order, "aligned": dst, "max_abs_logit_diff": diff})
    with open(os.path.join(d, "roles.json"), "w") as f:
        json.dump(roles, f, indent=1)
    return roles


if __name__ == "__main__":
    assert sys.argv[1] == "align"
    for m in sys.argv[2:]:
        r = align(m)
        print(name_of(m), r["assignment"], r["order_new_head_from_old"], r["max_abs_logit_diff"], flush=True)
