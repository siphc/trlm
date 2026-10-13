#!/bin/sh
cd /Users/svi/Desktop/trlm
for c in "$@"; do
  name=$(basename $c .py)
  OMP_NUM_THREADS=2 VOTE_FIT_THREADS=2 .venv/bin/python -m analysis.vote_fit analysis/candidates/$c.py --fit 24 > analysis/out/candidates/recall/vf_$name.log 2>&1
  echo "done $name"
done
