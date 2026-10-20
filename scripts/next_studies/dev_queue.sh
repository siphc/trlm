#!/bin/sh
# Development runs of studies B (training against the law) and A (fixed votes) on this machine's GPU, applying the
# pre-registered decision rules between steps (logs/prereg_law_pressure.md, logs/prereg_fixed_votes.md). Resumable:
# finished runs are skipped. Usage: scripts/next_studies/dev_queue.sh [device]   (default mps)
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
DEV=${1:-mps}
rule() { $PY -c "import json; print(json.load(open('analysis/out/$1/rules.json'))$2)"; }
echo "== study B: R2 (step time) $(date)"
[ -n "$(rule law_pressure ".get('R2', {}).get('law_windows', '')" 2>/dev/null)" ] || $PY -m analysis.law_pressure.rules r2 --device $DEV || exit 1
LW=$(rule law_pressure "['R2']['law_windows']")
echo "== study B: development grid (law_windows $LW) $(date)"
$PY -m src.train.runs law_dev --device $DEV --law_windows $LW || exit 1
OMP_NUM_THREADS=2 $PY -m analysis.law_pressure.rules r1 || exit 1
LAM=$(rule law_pressure "['R1']['lambda_star']")
echo "== study B: arms attn and vote at lambda* $LAM $(date)"
$PY -m src.train.runs law_dev --device $DEV --arms attn,vote --law_lams "$LAM" --law_windows $LW || exit 1
OMP_NUM_THREADS=2 $PY -m analysis.law_pressure.rules r3 || exit 1
echo "== study A: step time on $DEV $(date)"
[ -n "$(rule fixed_votes ".get('R2_$DEV', {}).get('epoch_hours', '')" 2>/dev/null)" ] || $PY -m analysis.fixed_votes.rules r2 --device $DEV || exit 1
# R2 binds on the confirmatory hardware (the instance); development here runs unless a step exceeds 0.6 s (operational)
if [ "$($PY -c "import json; print(json.load(open('analysis/out/fixed_votes/rules.json'))['R2_$DEV']['seconds_per_step']['fixed_votes'] > 0.6)")" = "True" ]; then
  echo "study A: a step takes more than 0.6 s on $DEV; development runs left to the instance; stopping here"; exit 0
fi
echo "== study A: R1 (LR check) $(date)"
$PY -m src.train.runs fixed_lr --device $DEV || exit 1
$PY -m analysis.fixed_votes.rules r1 || exit 1
LR=$(rule fixed_votes "['R1']['peak_lr']")
echo "== study A: development runs base and shift, seed 0, LR $LR $(date)"
$PY -m src.train.runs fixed --name base --seeds 0 --peak_lr "$LR" --device $DEV || exit 1
$PY -m src.train.runs fixed --name shift --seeds 0 --peak_lr "$LR" --device $DEV || exit 1
echo "development queue done $(date)"
