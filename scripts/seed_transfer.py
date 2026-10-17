"""Phase 8 for the mechanism program (v1.2 onward): keep the code and corpus statistics, refit only the named scalars
to another seed's model, and record what had to change.

  .venv/bin/python scripts/seed_transfer.py --models attn_s1 attn_s2 attn_s3 attn_s4

Per model:
  1. Role assignment: every head rule (LOCAL-0, LOCAL-2, RECALL) is fitted to every model head's own attention
     pattern; the assignment of rules to distinct heads with the lowest total pattern KL wins
     (mech_seed_transfer.fit_assignment). The leftover head is unmodelled (head 3's role in seed 0).
  2. programs/<model>/ = program/ (without tests); its constants are refitted with scripts/fit_scalars.py
     (attention and votes alternated, as for seed 0) and written into the copy.
  3. Harness on dev_fast (no claims: the claim tests name seed 0's heads).
Writes analysis/out/<model>/seed_transfer.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def harness(dst, model, note):
    r = subprocess.run([sys.executable, "-m", "harness.eval_fidelity", "--program", dst, "--model", model,
                        "--split", "dev_fast", "--no_claims", "--note", note], cwd=REPO, capture_output=True,
                       text=True)
    lines = r.stdout.splitlines()
    start = max((i for i, l in enumerate(lines) if l == "{"), default=None)
    return json.loads("\n".join(lines[start:])) if start is not None else {"error": r.stderr[-3000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()
    from analysis.mech_eval import program
    from mech_seed_transfer import fit_assignment

    for model in args.models:
        out_dir = os.path.join(REPO, "analysis", "out", model)
        os.makedirs(out_dir, exist_ok=True)
        P = program()
        assignment, table = fit_assignment(model, P.STATS)
        print(model, "assignment:", assignment, flush=True)
        dst = os.path.join("programs", model)
        if os.path.exists(os.path.join(REPO, dst)):
            shutil.rmtree(os.path.join(REPO, dst))
        shutil.copytree(os.path.join(REPO, "program"), os.path.join(REPO, dst),
                        ignore=shutil.ignore_patterns("tests", "__pycache__"))
        assign = ",".join(f"{r}={h}" for r, h in assignment.items())
        r = subprocess.run([sys.executable, "scripts/fit_scalars.py", "--model", model, "--assign", assign,
                            "--iterations", str(args.iterations), "--write", dst], cwd=REPO, capture_output=True,
                           text=True)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        fitted = json.load(open(os.path.join(out_dir, "fit_scalars.json")))
        res = {"model": model, "assignment": {k: int(v) for k, v in assignment.items()},
               "assignment_fits": {f"{rl} -> head {h}": v for (rl, h), v in table.items()},
               "fit_scalars": fitted,
               "harness_dev_fast": harness(dst, model, f"mechanism v1.2 seed transfer ({assign})")}
        with open(os.path.join(out_dir, "seed_transfer.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        h = res["harness_dev_fast"]
        print(model, "dev_fast KL", h.get("kl"), "top1", h.get("top1"), flush=True)


if __name__ == "__main__":
    main()
