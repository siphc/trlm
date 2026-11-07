#!/bin/bash
# Ladder pilot (logs/ladder/design.md, Training): L1 then L5, seed 0, arm ln, on the laptop GPU.
#   nohup caffeinate -dimsu scripts/ladder/pilot_queue.sh > logs/runs/ladder_pilot.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for R in L1 L5; do
  $PY -m src.train.ladder_runs pilot_lr --rung $R --device mps
  $PY -m src.train.ladder_runs pilot_budget --rung $R --device mps
done
$PY -m src.train.ladder_runs pilot_report
echo PILOT_DONE
