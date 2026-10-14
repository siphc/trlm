"""One discovery-loop iteration: run the harness, then commit with the metrics in the message.

  .venv/bin/python scripts/iterate.py --model attn_s0 --msg "bigram: WORD only" [--split dev_fast] [--explore]

The harness appends the row to logs/iterations.csv; this script commits program/, logs/, scripts/ and the
already-tracked files of analysis/ (new analysis files only via --add) with the harness numbers. Failed harness runs are committed too (message says FAILED).
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--msg", required=True)
    ap.add_argument("--split", default="dev_fast")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--kind", default="iter", help="commit prefix: iter (fidelity), readability, curve, explore")
    ap.add_argument("--no_commit", action="store_true")
    ap.add_argument("--add", nargs="*", default=[], help="new (untracked) files under analysis/ to include")
    args = ap.parse_args()
    r = subprocess.run([sys.executable, "-m", "harness.eval_fidelity", "--model", args.model, "--split", args.split,
                        "--note", args.msg], cwd=REPO, capture_output=True, text=True)
    sys.stderr.write(r.stderr[-3000:])
    prefix = "explore: " if args.explore else f"{args.kind}: "
    if r.returncode != 0:
        title = f"{prefix}{args.msg} | FAILED (harness exit {r.returncode})"
        body = r.stderr[-1500:]
    else:
        lines = r.stdout.splitlines()  # pytest output from the claim tests precedes the harness's JSON
        start = max(i for i, line in enumerate(lines) if line == "{")
        m = json.loads("\n".join(lines[start:]))
        title = (f"{prefix}{args.msg} | {args.model} {args.split} KL {m['kl']:.4f} top1 {m['top1']:.3f} "
                 f"LR {m['loss_recovered']:.3f}")
        body = (f"kl {m['kl']} (sem {m['kl_sem']}), top1 {m['top1']}, loss_recovered {m['loss_recovered']}, "
                f"ce_program {m['ce_program']}, ce_model {m['ce_model']}, ce_bigram {m['ce_bigram']}\n"
                f"lint_ok {m['lint_ok']}, loc {m['loc']}, numeric_literals {m['numeric_literals']}, "
                f"largest_literal {m['largest_literal']}, claims {m['claims_passed']}/{m['claims_total']}\n"
                f"program_sha {m['program_sha']}, sandbox: {m['sandbox']}")
        print(json.dumps(m, indent=1))
    if args.no_commit:
        print(title)
        return
    # analysis/: only files already tracked, plus those named with --add (other agents may be working there)
    subprocess.run(["git", "add", "program", "logs", "scripts", "harness/selftest"], cwd=REPO)
    subprocess.run(["git", "add", "-u", "analysis"], cwd=REPO)
    if args.add:
        subprocess.run(["git", "add", "--"] + args.add, cwd=REPO)
    subprocess.run(["git", "commit", "-q", "-m", f"{title}\n\n{body}\n\n{TRAILER}"], cwd=REPO)
    print(title)


if __name__ == "__main__":
    main()
