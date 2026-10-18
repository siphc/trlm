#!/bin/sh
# Readable-training side study, phase 4 (confirmatory): arms O and I at the FINAL lambdas, seeds 1-4, one epoch each
# (MPS), then the measurement suite on each model as it finishes (CPU, nice 20, in the background). Resumable.
# Refuses to run unless the FINAL pre-registration is committed.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
git log --format=%s -- logs/prereg_readable_training.md | grep -q "FINAL pre-registration" || { echo "FINAL not committed"; exit 1; }
LO=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3full']['O']['lambda_final'])")
LI=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3full']['I']['lambda_final'])")
for seed in 1 2 3 4; do
  for arm in O I; do
    lam=$LO; [ $arm = I ] && lam=$LI
    $PY -m src.train.runs readable --arm $arm --lam "$lam" --seeds $seed --device mps || exit 1
    (OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite "models/readable/${arm}_l${lam}_s${seed}/final.pt" \
      --lock --harness --refit >> "logs/runs/readable_suite_${arm}_l${lam}_s${seed}.log" 2>&1 &)
  done
done
echo "confirmatory training done $(date)"
