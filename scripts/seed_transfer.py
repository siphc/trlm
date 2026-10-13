"""Phase 8: freeze the program's structure from seed 0 and refit it to another seed's model.

  .venv/bin/python scripts/seed_transfer.py --model attn_s1 [--skip_fit]

For the target model: dense fit with the frozen structure (analysis/run_fit.py), then the main-line pruning
recipe (analysis/prune.py, same flags as the current main line), then emit into programs/<model>/ (a copy of
program/'s code with its own tables), run the harness there (dev_fast), and write a comparison of the refitted
tables with seed 0's into analysis/out/<model>/transfer.json. Only tables change; the code is identical.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")
# Main-line recipe (v10). Keep in sync with the latest adopted pipeline.
DENSE = ["--n_heads", "4", "--attn_steps", "800", "--class_steps", "1000", "--steps", "3000", "--refit_steps", "0",
         "--n_total", "40000", "--n_windows", "8000", "--lr", "0.02", "--save_dense", "--no_emit"]
PRUNE = ["--target", "60000", "--rounds", "16", "--refit_steps", "400", "--final_refit_steps", "2000", "--rank",
         "fisher", "--nonredundant", "--recenter", "--importance_batches", "150"]


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)} (see {log})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip_fit", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    m = args.model
    out = os.path.join(REPO, "analysis", "out", m)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(REPO, "logs", f"seed_transfer_{m}.log")
    dense = os.path.join(out, "student_transfer_dense_dense.pt")
    final = os.path.join(out, "student_transfer_gp60k.pt")
    if not args.skip_fit:
        if not os.path.exists(os.path.join(out, "mean_dists.npz")):
            run([PY, "-m", "analysis.mean_dists", "--model", m, "--n", "4000", "--device", args.device], log)
        run([PY, "-m", "analysis.run_fit", "--model", m, "--tag", "transfer_dense", "--device", args.device] + DENSE, log)
        run([PY, "-m", "analysis.prune", "--init", dense, "--model", m, "--tag", "transfer_gp60k", "--device",
             args.device] + PRUNE, log)
        os.remove(dense)  # 160 MB; the sparse final student is kept
    # program copy with this model's tables
    pdir = os.path.join(REPO, "programs", m)
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    shutil.copytree(os.path.join(REPO, "program"), pdir, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    code = ("import sys, torch; sys.path.insert(0, %r)\n"
            "from analysis.student import load_student\nfrom analysis.emit_program import emit\n"
            "from analysis.fitlib import train_windows\ntorch.set_num_threads(4)\n"
            "st = load_student(%r)\n"
            "emit(st, %r, 'Seed transfer: frozen seed-0 structure refitted to %s.', out_dir=%r, decimals=1, "
            "readable=True, windows=torch.as_tensor(train_windows(640, seed=3000)))\n") % (REPO, final, m, m, pdir)
    run([PY, "-c", code], log)
    r = subprocess.run([PY, "-m", "harness.eval_fidelity", "--program", pdir, "--model", m, "--split", "dev_fast",
                        "--no_claims", "--note", f"seed transfer: v10 structure refitted to {m}"],
                       cwd=REPO, capture_output=True, text=True)
    lines = r.stdout.splitlines()
    res = json.loads("\n".join(lines[max(i for i, l in enumerate(lines) if l == "{"):])) if r.returncode == 0 else {
        "error": r.stderr[-2000:]}
    json.dump(res, open(os.path.join(out, "transfer.json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("kl", "top1", "loss_recovered", "numeric_literals", "error")}))


if __name__ == "__main__":
    main()
