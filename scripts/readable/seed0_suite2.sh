#!/bin/sh
# After the stepped-down arm-I seed-0 run: its full suite; and the refit re-score for I_l1_s0 (started before the fix).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
while pgrep -f "scripts/readable/seed0_suite.sh" > /dev/null; do sleep 60; done
OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite models/readable/I_l1_s0/final.pt --refit >> logs/runs/readable_suite_I_l1_s0.log 2>&1
LI=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3full']['I']['lambda_final'])")
until [ -f "models/readable/I_l${LI}_s0/result.json" ] && ! pgrep -f "runs readable --arm I --lam ${LI} --seeds 0" > /dev/null; do sleep 60; done
OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite "models/readable/I_l${LI}_s0/final.pt" --lock --harness --refit >> "logs/runs/readable_suite_I_l${LI}_s0.log" 2>&1
echo "seed-0 suites (2) done $(date)"
