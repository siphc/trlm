#!/bin/bash
# Rungs L4-L7, arm ln, seeds 0-4 (logs/ladder_accounts/design.md). Usage: rungs_L4_L7.sh cpu "L4 L5" | rungs_L4_L7.sh mps "L6 L7"
set -e
cd "$(dirname "$0")/../.."
DEV=$1; RUNGS=$2
for R in $RUNGS; do
  OMP_NUM_THREADS=4 .venv/bin/python -m src.train.ladder_runs rung --rung $R --arm ln --device $DEV
done
echo RUNGS_DONE
