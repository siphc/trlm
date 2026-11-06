#!/bin/sh
# Study B confirmatory runs on this machine's GPU (decision 13; logs/prereg_law_pressure.md FINAL). Per seed 1-4: the
# twin (lambda 0), the penalised model (arm both, lambda* = 1), then both continued without the penalty. One seed is
# complete before the next starts. Resumable: re-running skips finished runs and resumes the current one.
# Usage: scripts/law_pressure/confirm_queue.sh [device]   (default mps)
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
DEV=${1:-mps}
for s in 1 2 3 4; do
  for lam in 0 1; do
    echo "== seed $s: arm both lambda $lam $(date)"
    $PY -m src.train.runs law --arm both --lam $lam --seeds $s --law_windows 16 --device $DEV || exit 1
  done
  for lam in 0 1; do
    echo "== seed $s: continuation of both_l${lam}_s$s $(date)"
    $PY -m src.train.runs law_restore --from models/law_pressure/both_l${lam}_s$s --device $DEV || exit 1
  done
done
echo "study B confirmatory training done $(date)"
