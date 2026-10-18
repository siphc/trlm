#!/bin/sh
# Readable-training side study, phase 3b: after the development grid, apply rules R3 and R5, then the seed-0 one-epoch
# runs at lambda* (arms O and I), the exploratory arm-I extras (attention only, votes only) at 100M tokens, and rules
# R3 (second part) and R7. Resumable.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
while pgrep -f "scripts/readable/dev_queue.sh" > /dev/null; do sleep 60; done
$PY -m analysis.readable_training.apply_rules r3 || exit 1
$PY -m analysis.readable_training.apply_rules r5 || exit 1
LO=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3']['O']['lambda_star'])")
LI=$($PY -c "import json; print(json.load(open('analysis/out/readable_training/rules.json'))['R3']['I']['lambda_star'])")
echo "lambda* O=$LO I=$LI"
$PY -m src.train.runs readable --arm O --lam "$LO" --seeds 0 --device mps && \
$PY -m src.train.runs readable --arm I --lam "$LI" --seeds 0 --device mps && \
$PY -m analysis.readable_training.apply_rules r3full && \
$PY -m analysis.readable_training.apply_rules r7 && \
$PY -m src.train.runs readable_dev --arms "" --extra I_att,I_vote --lam "$LI" --device mps
echo "phase 3b done $(date)"
