#!/bin/sh
# Step 5.4 (first half) of the deep study: 200M-token passes on the confirmatory models (MPS, sequential), started as
# soon as the calibration passes are done. Passes build tables only; arm fits on these models start only after
# logs/prereg_deep_look_write.md is committed (run scripts/confirmatory_fits.sh then).
cd "$(dirname "$0")/.." || exit 1
until [ -f analysis/out/look_write_deep/attn_s4/bgu_meta.json ] && ! pgrep -f "pass_bgu attn_s" > /dev/null; do sleep 60; done
for m in attn2L_s1 attn2L_s2 attn2L_s3 attn2L_s4 attn4L_s1 attn4L_s2 attn4L_s3 attn4L_s4; do
  echo "== pass $m $(date)"
  OMP_NUM_THREADS=4 .venv/bin/python -m analysis.look_write_deep.pass_bgu $m --tokens 200000000 --device mps || exit 1
done
echo "== confirmatory passes done $(date)"
