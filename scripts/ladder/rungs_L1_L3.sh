#!/bin/bash
# Rungs L1-L3 in both LayerNorm arms, seeds 0-4 (logs/ladder_accounts/design.md), on the CPU (faster than MPS at this size).
set -e
cd "$(dirname "$0")/../.."
for R in L1 L2 L3; do for A in ln noln; do
  OMP_NUM_THREADS=4 .venv/bin/python -m src.train.ladder_runs rung --rung $R --arm $A --device cpu
done; done
echo RUNGS_DONE
