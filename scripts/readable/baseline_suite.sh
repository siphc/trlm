#!/bin/sh
# Readable-training side study, phase 2: the measurement suite on the untreated models attn_s0..s4 (CPU, nice 20, two
# lanes). Refits only for the confirmatory twins attn_s1..s4. Resumable (finished steps are skipped).
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python"
lane() {
  for m in "$@"; do
    case $m in attn_s0) extra="";; *) extra="--refit";; esac
    OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite $m --lock --harness $extra \
      >> logs/runs/readable_baseline_suite_$m.log 2>&1
  done
}
lane attn_s1 attn_s3 attn_s0 &
lane attn_s2 attn_s4 &
wait
echo "baseline suite done $(date)"
