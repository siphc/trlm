"""The measurement suite for one model (pre-registration: logs/prereg_readable_training.md, "Tools").

  .venv/bin/python -m analysis.readable_training.suite <model id or final.pt> [--seed S] [--harness] [--lock] [--refit]

Steps (each saved into analysis/out/readable_training/<name>/suite.json as it finishes; finished steps are skipped):
  align     head roles and the head-aligned checkpoint (models.py)
  fidelity  KL to program v1.3 on train_eval, dev_fast and the out-of-distribution sets: the twin's probe samples
            (probe_attn_s{seed}), program samples, sentence- and word-shuffled dev windows (tools b1-check, c)
  internals pattern KL to the rules (d3, also out of distribution), vote correlations (d4), ablation shifts (d5)
  lock      (--lock) tables_vs_attention arms D and S{0,1,2}-joint on the aligned checkpoint (d2)
  harness   (--harness) harness dev_fast with claim tests on the aligned checkpoint (b1, d1; logged in iterations.csv)
  refit     (--refit) all program scalars refitted to the model (one round), then the harness without claims (b2)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from analysis.readable_training import models as M
from analysis.readable_training.common import OUT, REPO

OOD = ["program_samples", "sent_shuffle", "word_shuffle"]          # plus probe_attn_s{seed}


def harness(model, program_dir="program", claims=True, note=""):
    cmd = [sys.executable, "-m", "harness.eval_fidelity", "--program", program_dir, "--model", model,
           "--split", "dev_fast", "--note", note] + ([] if claims else ["--no_claims"])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    lines = r.stdout.splitlines()
    start = max((i for i, l in enumerate(lines) if l == "{"), default=None)
    if start is None:
        return {"error": r.stderr[-3000:]}
    return json.loads("\n".join(lines[start:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, default=None, help="the twin's seed (default: from the name's _s<k>)")
    ap.add_argument("--harness", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    name = M.name_of(a.model)
    seed = a.seed if a.seed is not None else int(re.search(r"_s(\d)", name).group(1))
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "suite.json")
    res = json.load(open(path)) if os.path.exists(path) else {"model": a.model, "name": name, "seed": seed}

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1)

    t0 = time.time()
    if "align" not in res:
        res["align"] = M.align(a.model); save()
    aligned = res["align"]["aligned"]
    ood = [f"probe_attn_s{seed}"] + OOD
    if "fidelity" not in res:
        from analysis.readable_training import fidelity

        res["fidelity"] = fidelity.run(aligned, ["train_eval", "dev_fast"] + ood); save()
    if "internals" not in res:
        from analysis.readable_training import internals

        res["internals"] = internals.run(aligned, ood); save()
    if a.lock and "lock" not in res:
        r = subprocess.run([sys.executable, "-m", "analysis.token_geometry.tables_vs_attention", "--model", aligned,
                            "--arms", "D,S0-joint,S1-joint,S2-joint", "--tag", "_lock", "--out", d],
                           cwd=REPO, capture_output=True, text=True)
        tv = json.load(open(os.path.join(d, "tables_vs_attention_lock.json")))
        kl = {k: v["kl"] for k, v in tv["arms"].items()}
        res["lock"] = {"kl": kl, "L": sum(kl[f"S{h}-joint"] - kl["D"] for h in range(3)) / 3}; save()
    if a.harness and "harness" not in res:
        res["harness"] = harness(aligned, note=f"readable side study: {name} (aligned)"); save()
    if a.refit and "kl" not in res.get("refit", {}).get("harness", {}):
        # the harness sandbox cannot read analysis/, so refitted program copies live in programs/readable/ (not in git),
        # under a name that is a valid module name
        prog = os.path.join(REPO, "programs", "readable", "prog_" + re.sub(r"\W", "_", name))
        old = os.path.join(d, "program_refit")
        os.makedirs(os.path.dirname(prog), exist_ok=True)
        if "refit" in res and os.path.isdir(old) and not os.path.isdir(prog):
            shutil.move(old, prog)                                   # the fit is done; only the harness failed
        if not os.path.isdir(prog):
            shutil.copytree(os.path.join(REPO, "program"), prog, ignore=shutil.ignore_patterns("tests", "__pycache__"))
            subprocess.run([sys.executable, "scripts/fit_scalars.py", "--model", aligned,
                            "--assign", "head0=0,head2=2,head1=1", "--iterations", "1", "--write", prog,
                            "--out_dir", d], cwd=REPO, capture_output=True, text=True)
        h = harness(aligned, program_dir=prog, claims=False, note=f"readable side study: {name} refitted program")
        res["refit"] = {"fit": json.load(open(os.path.join(d, "fit_scalars.json")))["history"], "harness": h,
                        "program": os.path.relpath(prog, REPO)}
        save()
    res["seconds_last_run"] = time.time() - t0
    save()
    print(json.dumps({"name": name, "dev_fast_kl": res["fidelity"]["dev_fast"]["kl_model_prog"],
                      "pattern_kl": res["internals"]["d3_pattern"]["mean_kl_model_rule"],
                      "vote_rho": res["internals"]["d4_vote"]["mean_uniform"],
                      "lock": res.get("lock", {}).get("L"), "harness_kl": res.get("harness", {}).get("kl")}))


if __name__ == "__main__":
    main()
