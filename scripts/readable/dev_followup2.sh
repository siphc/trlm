#!/bin/sh
# Readable-training side study, phase 3b continued: R3's second part stepped arm I down (lambda 1 cost 3.7% at one
# epoch); repeat the seed-0 run at the new lambda, re-check R3, then R7 and the arm-I extras. Resumable.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
get() { $PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3full']['$1']['lambda_final'])"; }
LI=$(get I)
$PY -m src.train.runs readable --arm I --lam "$LI" --seeds 0 --device mps || exit 1
$PY -m analysis.readable_training.apply_rules r3full || exit 1
LI2=$(get I)
if [ "$LI2" != "$LI" ]; then
  $PY -m src.train.runs readable --arm I --lam "$LI2" --seeds 0 --device mps || exit 1
  $PY -m analysis.readable_training.apply_rules r3full || exit 1
  LI2=$(get I)
fi
$PY -m analysis.readable_training.apply_rules r7
$PY -m src.train.runs readable_dev --arms "" --extra I_att,I_vote --lam "$LI2" --device mps
echo "phase 3b (continued) done $(date)"
