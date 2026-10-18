#!/bin/sh
# Readable-training side study: the measurement suite on the seed-0 one-epoch models as soon as each finishes
# (development only; CPU, nice 20).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LO=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3']['O']['lambda_star'])")
LI=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3']['I']['lambda_star'])")
for run in "O_l${LO}_s0" "I_l${LI}_s0"; do
  until [ -f "models/readable/$run/result.json" ]; do sleep 60; done
  OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite "models/readable/$run/final.pt" --lock --harness \
    --refit >> "logs/runs/readable_suite_$run.log" 2>&1
done
echo "seed-0 suites done $(date)"
