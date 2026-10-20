"""Write the job files for a study's confirmatory phase on the instance (run_jobs.py format). Refuses to run before
the study's FINAL pre-registration is committed. Values come from the study's rules.json (development rules).

    .venv/bin/python scripts/aws/make_jobs.py A    # -> scripts/aws/jobs/A_train.txt, A_analysis.txt
    .venv/bin/python scripts/aws/make_jobs.py B    # -> B_train.txt, B_wave2.txt (continuations + law tools), B_wave3.txt

Waves run one after another (each needs the previous one's models).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = ".venv/bin/python"
SEEDS = [1, 2, 3, 4]
PREREG = {"A": "logs/prereg_fixed_votes.md", "B": "logs/prereg_law_pressure.md"}
STUDY = {"A": "fixed_votes", "B": "law_pressure"}


def final_committed(path):
    log = subprocess.run(["git", "log", "--format=%s", "--", path], cwd=REPO, capture_output=True, text=True).stdout
    return "FINAL pre-registration" in log


def write(name, lines):
    d = os.path.join(REPO, "scripts", "aws", "jobs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"scripts/aws/jobs/{name}: {len(lines)} jobs")


def law_tools(model):
    return f"{PY} -m analysis.look_write.uov_tables {model} && {PY} -m analysis.look_write.fit_arms {model}"


def main(study):
    if not final_committed(PREREG[study]):
        raise SystemExit(f"the FINAL pre-registration of study {study} ({PREREG[study]}) is not committed")
    rules = json.load(open(os.path.join(REPO, "analysis", "out", STUDY[study], "rules.json")))
    if study == "A":
        lr = rules["R1"]["peak_lr"]
        write("A_train.txt", [f"A_{n}_s{s} {PY} -m src.train.runs fixed --name {n} --seeds {s} --peak_lr {lr:g} --device cuda"
                              for n in ("base", "shift") for s in SEEDS])
        write("A_analysis.txt", [f"A_law_{n}_s{s} {PY} -m analysis.fixed_votes.law models/fixed_votes/{n}_s{s}/final.pt"
                                 for n in ("base", "shift") for s in SEEDS])
    else:
        lam, lw = rules["R1"]["lambda_star"], rules["R2"]["law_windows"]
        arms = ["both"] + [a for a in ("attn", "vote") if rules.get("R3", {}).get(a, {}).get("admitted")]
        train = [f"B_twin_s{s} {PY} -m src.train.runs law --arm both --lam 0 --seeds {s} --law_windows {lw} --device cuda"
                 for s in SEEDS]
        train += [f"B_{a}_s{s} {PY} -m src.train.runs law --arm {a} --lam {lam:g} --seeds {s} --law_windows {lw} --device cuda"
                  for a in arms for s in SEEDS]
        write("B_train.txt", train)
        runs = [f"both_l0_s{s}" for s in SEEDS] + [f"{a}_l{lam:g}_s{s}" for a in arms for s in SEEDS]
        wave2 = [f"B_restore_{r} {PY} -m src.train.runs law_restore --from models/law_pressure/{r} --device cuda"
                 for r in runs if r.startswith(("both_l0", f"both_l{lam:g}"))]
        wave2 += [f"B_law_{r} {law_tools(f'models/law_pressure/{r}/final.pt')}" for r in runs]
        write("B_wave2.txt", wave2)
        write("B_wave3.txt", [f"B_law_restore_{r} {law_tools(f'models/law_pressure/restore/{r}/final.pt')}"
                              for r in runs if r.startswith(("both_l0", f"both_l{lam:g}"))])


if __name__ == "__main__":
    main(sys.argv[1])
