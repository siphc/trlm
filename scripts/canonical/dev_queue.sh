#!/bin/sh
# Study C part B development (logs/study_C_readable/prereg_canonical_addendum.md, rule B1): on seed 5, one epoch each (MPS): the
# lambda = 0 run, then the controls Octl and Ictl at lambda 0.3 and 1; rule B1 (analysis/canonical/choose_lambda.py)
# asks for at most one more grid value per control. Waits until study B's queue has left the GPU. Resumable.
# Refuses to run before the human's G0/G1 approval is recorded in the addendum.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
grep -q "^G0/G1 approved" logs/study_C_readable/prereg_canonical_addendum.md || { echo "G0/G1 not recorded"; exit 1; }
while pgrep -f "law_pressure/confirm_queue.sh" > /dev/null; do sleep 300; done
$PY -m src.train.runs canonical_dev --arms none --seeds 5 --device mps || exit 1
$PY -m src.train.runs canonical_dev --arms Octl,Ictl --lams 0.3,1 --seeds 5 --device mps || exit 1
for round in 1 2 3; do
  NEED=$($PY -m analysis.canonical.choose_lambda | grep '^NEED' | head -1)
  [ -z "$NEED" ] && break
  set -- $NEED
  echo "== rule B1 asks for $2 at lambda $3 $(date)"
  $PY -m src.train.runs canonical_dev --arms "$2" --lams "$3" --seeds 5 --device mps || exit 1
done
$PY -m analysis.canonical.choose_lambda
echo "part B development done $(date)"
