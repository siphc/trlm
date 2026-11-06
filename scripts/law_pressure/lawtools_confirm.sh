#!/bin/sh
# Goal 1's law tools (usefulness tables, then the arms fit) on each of study B's 16 confirmatory models as it finishes.
# CPU, low priority. Skips models already done; exits if the training queue stops before a model exists.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
run() {
  d=models/law_pressure/$1
  until [ -f $d/result.json ]; do pgrep -f law_pressure/confirm_queue.sh > /dev/null || [ -f $d/result.json ] || exit 0; sleep 120; done
  [ -f analysis/out/look_write/$(echo "law_pressure/$1" | tr / _)/arms.json ] && return 0
  echo "== law tools $1 $(date)"
  OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.look_write.uov_tables $d/final.pt && \
  OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.look_write.fit_arms $d/final.pt
}
for s in 1 2 3 4; do for m in both_l0_s$s both_l1_s$s restore/both_l0_s$s restore/both_l1_s$s; do run $m; done; done
echo "study B confirmatory law tools done $(date)"
