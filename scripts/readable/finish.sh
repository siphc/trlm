#!/bin/sh
# After the confirmatory queue: wait until all eight suites have their refit harness score, then tool e (seed
# distances, names readable_<arm>_l0.3_s<k> as verdicts.py expects) and the pre-registered verdicts
# (output logs/runs/readable_verdicts.out, analysis/out/readable_training/verdicts.json).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
count() {
  $PY -c "
import json, os
n = 0
for a in 'OI':
    for s in range(1, 5):
        p = f'analysis/out/readable_training/readable_{a}_l0.3_s{s}/suite.json'
        n += os.path.exists(p) and 'kl' in json.load(open(p)).get('refit', {}).get('harness', {})
print(n)"
}
until [ "$(count)" = 8 ]; do sleep 300; done
specs=""
for a in O I; do for s in 1 2 3 4; do specs="$specs readable_${a}_l0.3_s$s=models/readable/${a}_l0.3_s$s/final.pt"; done; done
OMP_NUM_THREADS=2 nice -n 10 $PY -m analysis.readable_training.seed_distances $specs > logs/runs/readable_seed_distances.out 2>&1
$PY -m analysis.readable_training.verdicts > logs/runs/readable_verdicts.out 2>&1
echo "readable verdicts done $(date)" >> logs/runs/readable_verdicts.out
