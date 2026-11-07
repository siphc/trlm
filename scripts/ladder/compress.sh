#!/bin/bash
# Exploratory compression search (analysis/ladder/compress_probe.py), seed 0, CPU.
#   scripts/ladder/compress.sh "L1 L2 L3" "ln noln"     (rungs, arms; default arms ln)
cd "$(dirname "$0")/../.."
for R in $1; do for A in ${2:-ln}; do
  echo "=== $R $A"
  OMP_NUM_THREADS=${THREADS:-3} .venv/bin/python -m analysis.ladder.compress_probe --rung $R --arm $A --seed 0
done; done
echo COMPRESS_DONE
