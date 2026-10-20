#!/bin/sh
# Goal 1's law tools (usefulness tables, then the arms fit) on each of study B's development models as it finishes:
# the grid (arm both, lambda 1..100), then arms attn and vote at lambda*. CPU, low priority. Skips models already done.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
run() {
  d=models/law_pressure/dev/$1
  until [ -f $d/result.json ]; do pgrep -f dev_queue.sh > /dev/null || [ -f $d/result.json ] || return 0; sleep 120; done
  [ -f analysis/out/look_write/law_pressure_dev_$1/arms.json ] && return 0
  OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.look_write.uov_tables $d/final.pt && \
  OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.look_write.fit_arms $d/final.pt
}
for lam in 1 3 10 30 100; do run both_l${lam}_s0_100M; done
until grep -q '"R1"' analysis/out/law_pressure/rules.json 2>/dev/null; do pgrep -f dev_queue.sh > /dev/null || exit 0; sleep 120; done
LAM=$($PY -c "import json; print(json.load(open('analysis/out/law_pressure/rules.json'))['R1']['lambda_star'])")
for arm in attn vote; do run ${arm}_l${LAM}_s0_100M; done
echo "development law tools done $(date)"
