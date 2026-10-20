#!/bin/sh
# Copy the studies' results from the instance back to the laptop. The instance's rules.json files arrive as
# rules_from_instance.json (the laptop's own rules.json stay authoritative; merge by hand).
set -e
cd "$(dirname "$0")/../.."
. scripts/aws/common.sh
R="rsync -az -e \"$SSH\""
for d in models/law_pressure models/fixed_votes logs/runs/aws; do
  mkdir -p "$d"; eval $R --exclude 'ckpt_step*.pt' --exclude latest.pt "$HOST:trlm/$d/" "$d/"
done
for s in law_pressure fixed_votes; do
  mkdir -p "analysis/out/$s"
  eval $R --exclude rules.json --exclude '*.npy' "$HOST:trlm/analysis/out/$s/" "analysis/out/$s/"
  eval $R "$HOST:trlm/analysis/out/$s/rules.json" "analysis/out/$s/rules_from_instance.json" || true
done
eval $R --include 'law_pressure_*/***' --include 'fixed_votes_*/***' --exclude '*' "$HOST:trlm/analysis/out/look_write/" analysis/out/look_write/
echo "pulled from $HOST"
