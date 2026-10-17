#!/bin/sh
# Step 5.2 of the deep study: 200M-token passes on the one-layer models (MPS, sequential), each followed by its arm
# fits with both readouts (CPU, in the background, lowest priority), so passes and fits overlap.
cd "$(dirname "$0")/.." || exit 1
OUT=analysis/out/look_write_deep
for m in attn_s1 attn_s2 attn_s3 attn_s4; do
  echo "== pass $m $(date)"
  OMP_NUM_THREADS=4 .venv/bin/python -m analysis.look_write_deep.pass_bgu $m --tokens 200000000 --device mps || exit 1
  ( for p in ltotal total; do
      [ -f $OUT/$m/arms_$p.json ] || OMP_NUM_THREADS=2 nice -n 20 .venv/bin/python -m analysis.look_write_deep.fit_arms_deep $m --primary $p
    done ) >> logs/calibration_fits.log 2>&1 &
done
wait
echo "== calibration done $(date)"
