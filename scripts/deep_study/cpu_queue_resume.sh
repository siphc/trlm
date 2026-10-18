#!/bin/sh
# CPU analysis of the deep study while the training queue uses MPS (one process at a time, lowest priority,
# 2 threads; finished work is skipped). The echo test (addendum D) runs on MPS before training, not here.
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2
PY="nice -n 20 .venv/bin/python"
OUT=analysis/out/look_write_deep
echo "== growth test (addendum 2) $(date)"
[ -f $OUT/growth.json ] || $PY -m analysis.look_write_deep.growth
echo "== capacity: one-layer models and development seeds $(date)"
for m in attn_s0 attn_s1 attn_s2 attn_s3 attn_s4 attn2L_s0 attn4L_s0; do
  [ -f $OUT/$m/capacity.json ] || $PY -m analysis.look_write_deep.capacity $m
done
echo "== done $(date)"
