"""Run shell commands from a job file concurrently (on the instance). One job per line: "<name> <command...>".
Each job's output goes to logs/runs/aws/<name>.log; a job whose log ends with "JOB OK" is skipped on re-runs (the
training commands themselves also skip finished runs). Blank lines and lines starting with # are ignored.

    nohup .venv/bin/python scripts/aws/run_jobs.py jobs.txt --parallel 6 > logs/runs/aws/run_jobs.out 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

LOG = os.path.join("logs", "runs", "aws")


def done(name):
    p = os.path.join(LOG, f"{name}.log")
    if not os.path.exists(p):
        return False
    with open(p, "rb") as f:
        f.seek(max(0, os.path.getsize(p) - 200))
        return b"JOB OK" in f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="OMP threads per job")
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    jobs = [l.split(None, 1) for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    todo = [(n, c.strip()) for n, c in jobs if not done(n)]
    running, failed = {}, []
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads)}
    while todo or running:
        while todo and len(running) < a.parallel:
            name, cmd = todo.pop(0)
            log = open(os.path.join(LOG, f"{name}.log"), "a")
            running[name] = (subprocess.Popen(f"{cmd} && echo 'JOB OK'", shell=True, stdout=log, stderr=log, env=env), log)
            print(f"{time.strftime('%H:%M:%S')} start {name}", flush=True)
        for name, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                del running[name]
                (failed.append(name) if p.returncode else None)
                print(f"{time.strftime('%H:%M:%S')} {'FAILED' if p.returncode else 'done'} {name}", flush=True)
        time.sleep(5)
    print("all jobs finished; failed:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
