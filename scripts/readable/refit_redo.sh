#!/bin/sh
# Re-score the refitted programs (tool b2) after the sandbox path fix; the fits themselves are reused.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
lane() { for m in "$@"; do OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite $m --refit >> logs/runs/readable_refit_redo.log 2>&1; done; }
lane attn_s1 attn_s3 models/readable/O_l0.3_s0/final.pt &
lane attn_s2 attn_s4 &
wait
echo "refit redo done $(date)"
