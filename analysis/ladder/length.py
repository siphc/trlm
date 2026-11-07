"""R1, description length (logs/ladder/design.md), and the account rules.

Account bits: the source with comments and docstrings removed (parsed and re-printed with ast.unparse, which also
normalises layout; numeric constants keep their value at the precision written), compressed with raw LZMA2 at
preset 9 | EXTREME (no container header), times 8.

Weights reference: every tensor quantised uniformly over its own [min, max] to b bits per weight, for the fewest b
in 1..16 with KL(model || quantised model) at most the P1 bar max(KL_floor, 0.01) on dev_full (decision 19; with
--bar 0.01 the original reference); its bits are the compressed codes
(uint8 or uint16), the per-tensor min and max as float32, and the compressed source of reference_forward.py.

Account rules (checked here, not by the harness): imports from the harness's stdlib allowlist and NumPy only; none of
the harness's forbidden names or NumPy file functions. So every number an account uses is written in its source
(or computed in setup() from the train stream), and R1 counts it.

    .venv/bin/python -m analysis.ladder.length --rung L1 --arm ln --seed 0 [--account ladder/L1_ln/account.py]

Size against the text reference (decisions 19, 21; P2, a pass condition until decision 22 removed it, now reported):
the account is compared with its bits are at most the TEXT reference's total bits: every weight
tensor written as an array literal in the account format (fmt below: no plus sign), at the fewest decimals per
tensor (lowered greedily from 4, largest tensor first) with KL(model || rounded model) at most the P1 bar, plus the
compressed reference_forward.py. The binary reference above is still reported.
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


MAX_INLINE_NUMBERS = 8   # numeric literals inside functions: at most this many numbers per list or tuple


def _numeric_arrays(name, val, out):
    if isinstance(val, dict):
        for k, v in val.items():
            _numeric_arrays(f"{name}[{k!r}]", v, out)
        return
    if isinstance(val, (list, tuple)):
        try:
            val = np.asarray(val)
        except ValueError:
            return
    if isinstance(val, np.ndarray) and val.dtype.kind in "fiub" and val.size > 1:
        out[name] = val


def table_errors(path: str, V: int, ctx: int, dims: tuple[int, ...]) -> list[str]:
    """Decision 22: every module-level numeric array is declared in AXES; axes are "symbol" (size V), "position"
    (at most ctx), "dim" (one of `dims`), "member" (symbol ids), a CATEGORIES list or a COMPONENTS list (each with a
    stated meaning); at most one "dim" axis per array. Numeric list literals inside functions hold at most
    MAX_INLINE_NUMBERS numbers. The module is executed without setup()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("account_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    axes = getattr(mod, "AXES", {})
    cats = getattr(mod, "CATEGORIES", {})
    comps = getattr(mod, "COMPONENTS", {})
    err = []
    for k, meaning in comps.items():
        if not isinstance(meaning, str) or not meaning.strip():
            err.append(f"component {k}: no stated meaning")
    arrays = {}
    for name, val in vars(mod).items():
        if name.startswith("_") or name in ("AXES", "CATEGORIES", "COMPONENTS", "CLAIMS") or callable(val):
            continue
        _numeric_arrays(name, val, arrays)
    for name, val in arrays.items():
        ax = axes.get(name)
        if ax is None:
            err.append(f"{name}: shape {val.shape} not declared in AXES")
            continue
        if len(ax) != val.ndim:
            err.append(f"{name}: AXES gives {len(ax)} axes, array has {val.ndim}")
            continue
        if list(ax).count("dim") > 1:
            err.append(f"{name}: more than one dim axis (a map between hidden spaces)")
        for a, n in zip(ax, val.shape):
            ok = {"symbol": n == V, "position": n <= ctx, "dim": n in dims, "member": True}.get(a)
            if ok is None:
                ok = (a in cats and n == len(cats[a])) or (a in comps and n == len(comps[a]))
                if a not in cats and a not in comps:
                    err.append(f"{name}: axis {a!r} is not symbol, position, dim, member, a category or a component")
                    continue
            if not ok:
                err.append(f"{name}: axis {a!r} has size {n}")
        if "member" in ax and not (val.dtype.kind in "iu" and val.min() >= 0 and val.max() < V):
            err.append(f"{name}: member axis must hold symbol ids")
    with open(path) as f:
        tree = ast.parse(f.read())
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.Lambda)):
            for n in ast.walk(fn):
                if isinstance(n, (ast.List, ast.Tuple)):
                    k = sum(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in ast.walk(n))
                    if k > MAX_INLINE_NUMBERS:
                        err.append(f"line {n.lineno}: {k} numbers in a literal inside a function")
    return err


def quantise(w: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(w.min()), float(w.max())
    levels = 2 ** b - 1
    codes = np.round((w - lo) / (hi - lo) * levels) if hi > lo else np.zeros_like(w)
    return codes.astype(np.uint8 if b <= 8 else np.uint16), lo + codes / levels * (hi - lo)


def p1_bar(rung: str, arm: str, seed: int) -> float:
    from analysis.ladder.eval import P1_MIN_THRESHOLD, floor_logprobs

    Wn = windows(rung, "dev_full")
    fl = floor_logprobs(rung, arm, seed, Wn)
    assert fl is not None, "seed floor needs all five seeds trained"
    return max(float(kl_pos(model_logprobs(load(rung, arm, seed), Wn), fl).mean()), P1_MIN_THRESHOLD)


def weights_reference(rung: str, arm: str, seed: int, split: str = "dev_full", bar: float = MAX_KL) -> dict:
    model = load(rung, arm, seed)
    W = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    Wn = windows(rung, split)
    X = Wn[:, :-1].astype(np.int64)
    lp_m = model_logprobs(model, Wn)
    exact = forward(X[:50], W, model.cfg.n_heads)
    # checked against the model in float64 (float32 rounding alone exceeds 1e-4 on extreme log-probs, e.g. -71)
    lp_64 = model_logprobs(load(rung, arm, seed).double(), Wn[:50])
    assert np.abs(exact - lp_64).max() < 1e-4, "reference_forward does not reproduce the model"
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
        if kl <= bar:
            break
    ok = rows[-1]["kl"] <= bar
    n_params = sum(v.size for v in W.values())
    return {"bar": bar, "n_params": int(n_params), "code_bits": code_bits, "ladder": rows, "found": ok,
            "bits_per_weight": rows[-1]["bits_per_weight"] if ok else None,
            "total_bits": rows[-1]["weight_bits"] + code_bits if ok else None}


def fmt(x: float, p: int) -> str:
    s = f"{x:.{p}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def literal(v: np.ndarray, p: int) -> str:
    if v.ndim == 1:
        return "np.array([" + ", ".join(fmt(x, p) for x in v) + "])"
    return "np.array([\n" + "".join("    [" + ", ".join(fmt(x, p) for x in r) + "],\n" for r in v) + "])"


def text_reference(rung: str, arm: str, seed: int, bar: float, split: str = "dev_full") -> dict:
    model = load(rung, arm, seed)
    W = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    Wn = windows(rung, split)
    X = Wn[:, :-1].astype(np.int64)
    lp_m = model_logprobs(model, Wn)

    def kl(dec):
        Wr = {k: np.round(v, dec[k]) for k, v in W.items()}
        lq = np.concatenate([forward(X[i:i + 200], Wr, model.cfg.n_heads) for i in range(0, len(X), 200)])
        return float(kl_pos(lp_m, lq).mean())

    dec = {k: 4 for k in W}
    k_now = kl(dec)
    for k in sorted(W, key=lambda k: -W[k].size):
        while dec[k] > 0:
            trial = {**dec, k: dec[k] - 1}
            k_trial = kl(trial)
            if k_trial > bar:
                break
            dec, k_now = trial, k_trial
    src = "import numpy as np\nW = {}\n" + "".join(f"W[{k!r}] = {literal(np.round(v, dec[k]), dec[k])}\n"
                                                   for k, v in W.items())
    weight_bits = compressed_bits(strip_source(src).encode())
    code_bits = source_bits(os.path.join(os.path.dirname(__file__), "reference_forward.py"))
    return {"bar": bar, "decimals": dec, "kl": k_now, "found": k_now <= bar, "weight_bits": weight_bits,
            "code_bits": code_bits, "total_bits": weight_bits + code_bits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--account")
    ap.add_argument("--bar", type=float, help="KL bar for the reference (default: the P1 bar)")
    a = ap.parse_args()
    bar = a.bar if a.bar is not None else p1_bar(a.rung, a.arm, a.seed)
    ref = text_reference(a.rung, a.arm, a.seed, bar)
    out = {"rung": a.rung, "arm": a.arm, "seed": a.seed, "text_reference": ref,
           "binary_reference": weights_reference(a.rung, a.arm, a.seed, bar=bar)}
    if a.account:
        bits = source_bits(a.account)
        out["account"] = {"path": os.path.relpath(a.account, REPO), "bits": bits,
                          "rule_errors": rule_errors(a.account)}
        model = load(a.rung, a.arm, a.seed)
        out["account"]["table_errors"] = table_errors(a.account, model.cfg.vocab_size, model.cfg.context_length,
                                                      (model.cfg.d_model, model.cfg.d_head))
        out["size_vs_text_reference"] = bool(ref["found"] and bits <= ref["total_bits"])   # reported (decision 22)
        out["rules_pass"] = not out["account"]["rule_errors"] and not out["account"]["table_errors"]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
