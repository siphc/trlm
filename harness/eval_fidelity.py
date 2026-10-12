"""The only scorer. READ-ONLY for replacement agents.

Usage:
  python -m harness.eval_fidelity --model attn_s0 --split dev_fast
  python -m harness.eval_fidelity --model attn_s0 --split test --note "milestone 3"   (budgeted)

Steps:
  1. Lint the program (mandatory for program/).
  2. Start the program in a sandboxed subprocess (models/, data/ unreadable where the OS allows).
  3. Causality and validity checks on 3 windows.
  4. For every window: KL(model || program), top-1 agreement, cross-entropy against the data.
  5. Claim tests in <program>/tests/ (pytest) on held-out windows.
  6. Append a row to logs/iterations.csv (and logs/test_looks.csv for --split test).

Primary metric: mean per-token KL(model || program) in nats, over all 256 positions of every window.
Loss recovered = (L_bigram - L_program) / (L_bigram - L_model), cross-entropy against the data,
L_bigram from the train-split bigram baseline (data/baselines/bigram_logprobs.npy).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from harness.common import (CTX, REPO, ProgramError, ProgramRunner, bigram_logprobs, causality_check,
                            eval_windows, git_state, load_model, model_logprobs, program_sha,
                            validate_rows)
from harness.lint_program import lint_program

TEST_BUDGET = 10
ITER_FIELDS = ["timestamp", "commit", "dirty", "program_sha", "program", "model", "split", "n_windows",
               "kl", "kl_sem", "top1", "ce_program", "ce_model", "ce_bigram", "loss_recovered",
               "lint_ok", "loc", "numeric_literals", "largest_literal", "claims_passed", "claims_total",
               "sandbox", "seconds", "note"]


def run_claim_tests(program_dir: str, model_id: str) -> tuple[int, int]:
    tests = os.path.join(program_dir, "tests")
    if not os.path.isdir(tests) or not any(f.startswith("test_") for f in os.listdir(tests)):
        return 0, 0
    xml = os.path.join(REPO, "logs", ".claims_junit.xml")
    env = {**os.environ, "TRLM_MODEL": model_id, "TRLM_PROGRAM": os.path.abspath(program_dir),
           "PYTHONPATH": REPO}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", tests,
                    f"--junitxml={xml}"], cwd=REPO, env=env)
    if not os.path.exists(xml):
        return 0, 0
    root = ET.parse(xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = sum(int(s.get("tests", 0)) for s in suites)
    bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) for s in suites)
    os.remove(xml)
    return total - bad, total


def append_csv(path: str, fields: list[str], row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def evaluate(program_dir: str, model_id: str, split: str, max_windows: int | None = None,
             lint: bool = True, sandbox: bool = True, claims: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    lint_res = lint_program(program_dir) if lint else {"ok": None, "loc": None, "numeric_literals": None,
                                                       "largest_literal": None, "errors": []}
    if lint and not lint_res["ok"]:
        print("LINT FAILED:\n  " + "\n  ".join(lint_res["errors"]), file=sys.stderr)

    wins = eval_windows(split, model_id)
    if max_windows:
        wins = wins[:max_windows]
    model = load_model(model_id)
    big = bigram_logprobs()

    kl_sum, n_tok, top1, ce_p, ce_m, ce_b = 0.0, 0, 0, 0.0, 0.0, 0.0
    kl_per_window = []
    with ProgramRunner(program_dir, sandbox=sandbox) as runner:
        causality_check(runner, wins[:3], np.random.default_rng(0))
        bs = 25
        for i in range(0, len(wins), bs):
            chunk = wins[i : i + bs]
            lp_model = model_logprobs(model, chunk)
            for j, w in enumerate(chunk):
                x, y = w[:CTX].astype(int), w[1 : CTX + 1].astype(int)
                q = runner.predict(x.tolist())
                validate_rows(q, CTX)
                p_log = lp_model[j]
                p = np.exp(p_log)
                with np.errstate(invalid="ignore", divide="ignore"):
                    terms = np.where(p > 0, p * (p_log - q), 0.0)
                kl_w = terms.sum(axis=1)
                kl_per_window.append(kl_w.mean())
                kl_sum += kl_w.sum()
                n_tok += CTX
                top1 += int((p_log.argmax(1) == q.argmax(1)).sum())
                r = np.arange(CTX)
                ce_p += -q[r, y].sum()
                ce_m += -p_log[r, y].sum()
                ce_b += -big[x, y].sum()
            if verbose:
                print(f"  {min(i + bs, len(wins))}/{len(wins)} windows  KL so far {kl_sum / n_tok:.5f}",
                      file=sys.stderr, flush=True)

    kl = kl_sum / n_tok
    L_p, L_m, L_b = ce_p / n_tok, ce_m / n_tok, ce_b / n_tok
    passed, total = run_claim_tests(program_dir, model_id) if claims else (0, 0)
    commit, dirty = git_state()
    kpw = np.asarray(kl_per_window)
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "dirty": dirty, "program_sha": program_sha(program_dir),
        "program": os.path.relpath(os.path.abspath(program_dir), REPO),
        "model": os.path.basename(model_id.rstrip("/")) if not os.path.exists(model_id) else model_id,
        "split": split, "n_windows": len(wins),
        "kl": round(kl, 6), "kl_sem": round(float(kpw.std(ddof=1) / np.sqrt(len(kpw))) if len(kpw) > 1 else 0.0, 6),
        "top1": round(top1 / n_tok, 5),
        "ce_program": round(L_p, 5), "ce_model": round(L_m, 5), "ce_bigram": round(L_b, 5),
        "loss_recovered": round((L_b - L_p) / (L_b - L_m), 5),
        "lint_ok": lint_res["ok"], "loc": lint_res["loc"], "numeric_literals": lint_res["numeric_literals"],
        "largest_literal": lint_res["largest_literal"],
        "claims_passed": passed, "claims_total": total,
        "sandbox": runner.sandbox_status, "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="program")
    ap.add_argument("--model", required=True, help="model id (e.g. attn_s0) or path to a .pt checkpoint")
    ap.add_argument("--split", default="dev_fast", choices=["dev_fast", "dev_full", "probe", "test"])
    ap.add_argument("--note", default="")
    ap.add_argument("--max_windows", type=int, default=None, help="quick look; not logged")
    ap.add_argument("--no_lint", action="store_true", help="self-tests only; refused for program/")
    ap.add_argument("--no_sandbox", action="store_true", help="self-tests only; refused for program/")
    ap.add_argument("--no_claims", action="store_true")
    ap.add_argument("--no_log", action="store_true")
    args = ap.parse_args()

    is_main_program = os.path.abspath(args.program) == os.path.join(REPO, "program")
    if is_main_program and (args.no_lint or args.no_sandbox):
        sys.exit("--no_lint / --no_sandbox are not allowed for program/")
    if args.split == "test":
        if not args.note:
            sys.exit("--split test requires --note")
        looks = os.path.join(REPO, "logs", "test_looks.csv")
        model_key = os.path.basename(args.model.rstrip("/"))
        used = 0
        if os.path.exists(looks):
            with open(looks) as f:
                used = sum(1 for r in csv.DictReader(f) if r["model"] == model_key)
        if used >= TEST_BUDGET:
            sys.exit(f"test budget exhausted for {model_key} ({used}/{TEST_BUDGET})")
        if args.max_windows:
            sys.exit("--max_windows is not allowed with --split test")

    try:
        res = evaluate(args.program, args.model, args.split, args.max_windows,
                       lint=not args.no_lint, sandbox=not args.no_sandbox, claims=not args.no_claims)
    except ProgramError as e:
        print(f"PROGRAM ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    res["note"] = args.note
    print(json.dumps(res, indent=2))
    if args.max_windows or args.no_log:
        print("(not logged)", file=sys.stderr)
        return
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    append_csv(os.path.join(REPO, "logs", "iterations.csv"), ITER_FIELDS, res)
    if args.split == "test":
        append_csv(os.path.join(REPO, "logs", "test_looks.csv"), ITER_FIELDS, res)


if __name__ == "__main__":
    main()
