#!/bin/sh
# Study C part B confirmatory runs (logs/study_C_readable/prereg_canonical_addendum.md FINAL): per seed 1-4, one epoch each (MPS):
# arm O at lambda 1 (pressure curve), Octl and Ictl at their rule-B1 lambdas; the suite (align, fidelity, internals,
# harness) on each model as it finishes (CPU, nice 20, background). Then seed distances per treatment and
# analysis/canonical/convergence.py. Resumable. Refuses to run unless the FINAL addendum is committed.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
git log --follow --format=%s -- logs/study_C_readable/prereg_canonical_addendum.md | grep -q "FINAL" || { echo "FINAL addendum not committed"; exit 1; }
while pgrep -f "law_pressure/confirm_queue.sh|canonical/dev_queue.sh" > /dev/null; do sleep 300; done
LO=$($PY -c "import json; print('%g' % json.load(open('analysis/out/canonical/rules.json'))['B1']['Octl']['lambda'])")
LI=$($PY -c "import json; print('%g' % json.load(open('analysis/out/canonical/rules.json'))['B1']['Ictl']['lambda'])")
for seed in 1 2 3 4; do
  for spec in "O 1" "Octl $LO" "Ictl $LI"; do
    set -- $spec
    $PY -m src.train.runs readable --arm $1 --lam "$2" --seeds $seed --device mps || exit 1
    (OMP_NUM_THREADS=2 nice -n 20 $PY -m analysis.readable_training.suite "models/readable/${1}_l${2}_s${seed}/final.pt" \
      --harness >> "logs/runs/canonical_suite_${1}_l${2}_s${seed}.log" 2>&1 &)
  done
done
while pgrep -f "analysis.readable_training.suite" > /dev/null; do sleep 120; done
for spec in "O 1" "Octl $LO" "Ictl $LI"; do
  set -- $spec
  $PY -m analysis.readable_training.seed_distances $(for k in 1 2 3 4; do printf "readable_%s_l%s_s%s=models/readable/%s_l%s_s%s/final.pt " $1 $2 $k $1 $2 $k; done)
done
$PY -m analysis.canonical.convergence --treatments "O_l0.3,I_l0.3,O_l1,Octl_l$LO,Ictl_l$LI" \
  --compare "O_l0.3:Octl_l$LO,I_l0.3:Ictl_l$LI"
echo "part B confirmatory done $(date)"
