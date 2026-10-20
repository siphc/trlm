#!/bin/sh
# Study A's measurements (analysis/fixed_votes/law.py) on each development model as it finishes (CPU, low priority).
cd "$(dirname "$0")/../.." || exit 1
for n in base shift; do
  d=models/fixed_votes/${n}_s0
  until [ -f $d/result.json ]; do pgrep -f dev_queue.sh > /dev/null || [ -f $d/result.json ] || exit 0; sleep 120; done
  [ -f analysis/out/fixed_votes/${n}_s0/law.json ] || OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.fixed_votes.law $d/final.pt
done
echo "study A development law fits done $(date)"
