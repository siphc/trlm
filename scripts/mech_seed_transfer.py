"""Phase 8 for the mechanism program: keep its code and corpus statistics, refit only its named scalars to another
seed's model, and record what had to change.

  .venv/bin/python scripts/mech_seed_transfer.py --model attn_s1 [--no_harness]

Steps:
  1. Which model head plays which role? Every head rule of the program (LOCAL-0, LOCAL-2, RECALL) is fitted to
     every head of the target model (attention scalars only, pattern cross-entropy, 32 TRAIN windows); the
     assignment of rules to distinct heads with the lowest total pattern KL wins. The leftover head is unmodelled
     (it plays head 3's role in seed 0).
  2. Vote strengths are refitted to the target model with the assigned attention rules (convex; analysis/vote_fit).
  3. The program is copied to programs/<model>/ with only the constants changed, and run through the harness
     (dev_fast). analysis/out/<model>/mech_transfer.json records the assignment, every scalar before and after,
     and the KL numbers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ROLES = ["head0", "head2", "head1"]          # program modules with an attention rule (LOCAL-0, LOCAL-2, RECALL)


def fit_assignment(model_id, stats, n=32):
    import program.head0, program.head1, program.head2
    from analysis.attention_fit import fit_rule, model_attention, module_copy, pattern_ce
    from analysis.fitlib import train_windows

    mods = {"head0": program.head0, "head1": program.head1, "head2": program.head2}
    W = train_windows(n, seed=2024)
    A = model_attention(model_id, W)
    H = A.shape[1]
    table = {}
    for role in ROLES:
        for h in range(H):
            m = module_copy(mods[role], tag=f"{role}_h{h}")
            before = pattern_ce(m, A[:, h], W, stats)[1]
            fitted = fit_rule(m, A[:, h], W, stats, keep=True)
            table[(role, h)] = {"pattern_kl_seed0_scalars": float(before),
                                "pattern_kl": float(pattern_ce(m, A[:, h], W, stats)[1]), "scalars": fitted}
            print(f"{role} -> head {h}: pattern KL {before:.3f} -> {table[(role, h)]['pattern_kl']:.3f}", flush=True)
    best = min(itertools.permutations(range(H), len(ROLES)),
               key=lambda hs: sum(table[(r, h)]["pattern_kl"] for r, h in zip(ROLES, hs)))
    return dict(zip(ROLES, best)), table


def set_constants(mod, scalars):
    from analysis.attention_fit import RULE_PARAMS, _set

    spec = RULE_PARAMS[mod.__name__.split(".")[-1]]
    vals = []
    for attr, key, tr in spec:
        v = scalars[attr + ("" if key is None else f"[{key!r}]")]
        vals.append(np.log(v) if tr == "log" else v)
    _set(mod, spec, vals)


def refit_votes(model_id, P, n_fit=32, n_eval=64):
    """Refit the 25 vote strengths to the target model; returns {direction: strength} and held-out KLs."""
    from analysis import vote_fit
    from analysis.fitlib import train_windows
    from harness.common import load_model, model_logprobs

    model = load_model(model_id)
    Wf, We = train_windows(n_fit, seed=vote_fit.FIT_SEED), train_windows(n_eval, seed=2000)
    probs = np.exp(model_logprobs(model, Wf)).reshape(-1, 2000).astype(np.float32)
    X, keys, b0 = vote_fit.stacked(P, Wf, None)
    beta = vote_fit.fit(X, probs, [b0[k] for k in keys])
    del X
    lpe = model_logprobs(model, We)
    kls = vote_fit.evaluate(P, We, lpe, None, {"seed-0 strengths": (keys, [b0[k] for k in keys]),
                                               "refitted": (keys, beta)})
    return dict(zip(keys, map(float, beta))), {k: float(v.mean()) for k, v in kls.items()}


def strengths_to_constants(strengths):
    """Map vote directions back to the modules' named constants."""
    from program import head1

    c = {"follow": {"STRENGTH": strengths["follow.bigram"], "UNIGRAM": strengths["follow.unigram"]},
         "phrase": {"TRIGRAM": strengths["phrase.trigram"], "FOURGRAM": strengths["phrase.fourgram"]}}
    for m in ("head0", "head2"):
        c[m] = {"STRENGTH_SELF": strengths[f"{m}.self"], "STRENGTH_NEAR": strengths[f"{m}.near"],
                "STRENGTH_FAR": strengths[f"{m}.far"]}
    c["head1"] = {part: {g: strengths[f"head1.{part.lower()}.{g}"] for g in head1.GROUP_NAMES}
                  for part in ("COPY", "TOPIC", "NEXT")}
    return c


def fmt(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f'"{k}": {fmt(x)}' for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return f"{v:.3f}" if abs(v) >= 0.01 else f"{v:.4f}"


def write_program_copy(model_id, constants, note):
    """programs/<model>/: program/ with the listed module constants replaced (same code otherwise). Each constant's
    whole assignment statement is rewritten (located with ast, so multi-line dicts and tuple assignments work)."""
    import ast

    dst = os.path.join(REPO, "programs", model_id)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(REPO, "program"), dst, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for mod, consts in constants.items():
        path = os.path.join(dst, f"{mod}.py")
        lines = open(path).read().split("\n")
        edits = []
        for node in ast.parse("\n".join(lines)).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            names = [tgt.id] if isinstance(tgt, ast.Name) else [e.id for e in getattr(tgt, "elts", [])
                                                                  if isinstance(e, ast.Name)]
            if not any(n in consts for n in names):
                continue
            if isinstance(tgt, ast.Name):
                text = f"{tgt.id} = {fmt(consts[tgt.id])}  # {note}"
            else:
                old_vals = [ast.literal_eval(e) for e in node.value.elts]
                vals = [fmt(consts[n]) if n in consts else repr(v) for n, v in zip(names, old_vals)]
                text = f"{', '.join(names)} = {', '.join(vals)}  # {note}"
            edits.append((node.lineno - 1, node.end_lineno, text))
        missing = set(consts) - {n for node in ast.parse("\n".join(lines)).body if isinstance(node, ast.Assign)
                                 for t in node.targets for n in ([t.id] if isinstance(t, ast.Name) else
                                 [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)])}
        if missing:
            raise KeyError(f"{mod}: constants not found: {missing}")
        for a, b, text in sorted(edits, reverse=True):
            lines[a:b] = [text]
        open(path, "w").write("\n".join(lines))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--no_harness", action="store_true")
    args = ap.parse_args()
    from analysis.mech_eval import program

    P = program()
    assignment, table = fit_assignment(args.model, P.STATS)
    print("assignment:", assignment, flush=True)
    attention_constants = {}
    for role, h in assignment.items():
        mod = __import__(f"program.{role}", fromlist=[role])
        scalars = table[(role, h)]["scalars"]
        set_constants(mod, scalars)
        attention_constants[role] = {}
        for key, v in scalars.items():
            attr, _, idx = key.partition("[")
            if idx:
                cur = attention_constants[role].setdefault(attr, getattr(mod, attr))
                cur = list(cur) if isinstance(cur, (list, tuple)) else dict(cur)
                cur[eval(idx[:-1])] = v
                attention_constants[role][attr] = cur
            else:
                attention_constants[role][attr] = v
    strengths, kls = refit_votes(args.model, P)
    print("held-out KL:", kls, flush=True)
    constants = strengths_to_constants(strengths)
    for role, c in attention_constants.items():
        constants[role].update(c)
    dst = write_program_copy(args.model, constants, f"refitted to {args.model} (scripts/mech_seed_transfer.py)")
    out = {"model": args.model, "assignment": {r: int(h) for r, h in assignment.items()},
           "fits": {f"{r} -> head {h}": v for (r, h), v in table.items()},
           "constants": constants, "heldout_kl_train64": kls}
    if not args.no_harness:
        r = subprocess.run([sys.executable, "-m", "harness.eval_fidelity", "--program", dst, "--model", args.model,
                            "--split", "dev_fast", "--no_claims", "--note", "mechanism v1 seed transfer"],
                           cwd=REPO, capture_output=True, text=True)
        lines = r.stdout.splitlines()
        start = max((i for i, l in enumerate(lines) if l == "{"), default=None)
        out["harness"] = json.loads("\n".join(lines[start:])) if start is not None else r.stderr[-2000:]
    os.makedirs(os.path.join(REPO, "analysis", "out", args.model), exist_ok=True)
    with open(os.path.join(REPO, "analysis", "out", args.model, "mech_transfer.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("assignment", "heldout_kl_train64")}, indent=1))
    if "harness" in out and isinstance(out["harness"], dict):
        print("dev_fast KL", out["harness"]["kl"])


if __name__ == "__main__":
    main()
