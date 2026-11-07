"""R1, description length (logs/ladder/design.md), and the account rules.

Account bits: the source with comments and docstrings removed (parsed and re-printed with ast.unparse, which also
normalises layout; numeric constants keep their value at the precision written), compressed with raw LZMA2 at
preset 9 | EXTREME (no container header), times 8.

Weights reference: every tensor quantised uniformly over its own [min, max] to b bits per weight, for the fewest b
in 1..16 with KL(model || quantised model) <= 0.01 nats per symbol on dev_full; its bits are the compressed codes
(uint8 or uint16), the per-tensor min and max as float32, and the compressed source of reference_forward.py.

Account rules (checked here, not by the harness): imports from the harness's stdlib allowlist and NumPy only; none of
the harness's forbidden names or NumPy file functions. So every number an account uses is written in its source
(or computed in setup() from the train stream), and R1 counts it.

    .venv/bin/python -m analysis.ladder.length --rung L1 --arm ln --seed 0 [--account ladder/L1_ln/account.py]
"""
from __future__ import annotations

import argparse
import ast
import json
import lzma
import os

import numpy as np

from analysis.ladder.common import REPO, kl_pos, load, model_logprobs, windows
from analysis.ladder.reference_forward import forward
from harness.lint_program import ALLOWED_MODULES, FORBIDDEN_NAMES, FORBIDDEN_NP_ATTRS

FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
MAX_KL = 0.01


def compressed_bits(data: bytes) -> int:
    return 8 * len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=FILTERS))


def strip_source(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.unparse(tree)


def source_bits(path: str) -> int:
    with open(path) as f:
        return compressed_bits(strip_source(f.read()).encode())


def rule_errors(path: str) -> list[str]:
    with open(path) as f:
        tree = ast.parse(f.read())
    err = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods = [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [] if n.level else [(n.module or "").split(".")[0]]
        else:
            mods = []
        err += [f"line {n.lineno}: import {m}" for m in mods if m not in ALLOWED_MODULES]
        if isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            err.append(f"line {n.lineno}: {n.id}")
        if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_NP_ATTRS and isinstance(n.value, ast.Name) \
                and n.value.id in ("np", "numpy"):
            err.append(f"line {n.lineno}: np.{n.attr}")
    return err


def quantise(w: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(w.min()), float(w.max())
    levels = 2 ** b - 1
    codes = np.round((w - lo) / (hi - lo) * levels) if hi > lo else np.zeros_like(w)
    return codes.astype(np.uint8 if b <= 8 else np.uint16), lo + codes / levels * (hi - lo)


def weights_reference(rung: str, arm: str, seed: int, split: str = "dev_full") -> dict:
    model = load(rung, arm, seed)
    W = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    Wn = windows(rung, split)
    X = Wn[:, :-1].astype(np.int64)
    lp_m = model_logprobs(model, Wn)
    exact = forward(X[:50], W, model.cfg.n_heads)
    assert np.abs(exact - lp_m[:50]).max() < 1e-4, "reference_forward does not reproduce the model"
    code_bits = source_bits(os.path.join(os.path.dirname(__file__), "reference_forward.py"))
    rows = []
    for b in range(1, 17):
        q = {k: quantise(v, b) for k, v in W.items()}
        lp_q = np.concatenate([forward(X[i:i + 200], {k: v[1] for k, v in q.items()}, model.cfg.n_heads)
                               for i in range(0, len(X), 200)])
        kl = float(kl_pos(lp_m, lp_q).mean())
        payload = b"".join(c.tobytes() for c, _ in q.values()) + \
            np.array([[v.min(), v.max()] for v in W.values()], np.float32).tobytes()
        rows.append({"bits_per_weight": b, "kl": kl, "weight_bits": compressed_bits(payload)})
        if kl <= MAX_KL:
            break
    ok = rows[-1]["kl"] <= MAX_KL
    n_params = sum(v.size for v in W.values())
    return {"n_params": int(n_params), "code_bits": code_bits, "ladder": rows, "found": ok,
            "bits_per_weight": rows[-1]["bits_per_weight"] if ok else None,
            "total_bits": rows[-1]["weight_bits"] + code_bits if ok else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--account")
    a = ap.parse_args()
    out = {"rung": a.rung, "arm": a.arm, "seed": a.seed, "weights_reference": weights_reference(a.rung, a.arm, a.seed)}
    if a.account:
        out["account"] = {"path": os.path.relpath(a.account, REPO), "bits": source_bits(a.account),
                          "rule_errors": rule_errors(a.account)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
