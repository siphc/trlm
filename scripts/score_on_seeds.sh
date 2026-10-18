#!/bin/sh
# Score program/ unchanged (seed-0 constants) on other models with the harness (dev_fast, no claims: the claim tests
# name seed 0's heads). One harness row per model in logs/iterations.csv.
cd "$(dirname "$0")/.." || exit 1
for m in "$@"; do
  OMP_NUM_THREADS=3 .venv/bin/python -m harness.eval_fidelity --model "$m" --split dev_fast --no_claims \
    --note "v1.3 unchanged (seed-0 constants) on $m" > "logs/score_v13_$m.out" 2>&1
  grep -E '"kl"|"top1"|"loss_recovered"' "logs/score_v13_$m.out" | tr -d '\n'; echo " $m"
done
