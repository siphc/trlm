#!/bin/sh
# CPU-only analysis queue while the deep models train (one process at a time, 2 threads).
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2
PY=.venv/bin/python
echo "== B: paired SEs (regenerate shuffled tables, refit) $(date)"
nice -n 20 $PY -m analysis.look_write.uov_tables attn_s0 attn_s3 attn_s4 && \
nice -n 20 $PY -m analysis.look_write.fit_arms attn_s0 attn_s3 attn_s4
echo "== C: distance band 8..63 $(date)"
nice -n 20 $PY -m analysis.look_write.uov_tables attn_s1 attn_s2 attn_s3 attn_s4 attn_s0 --band 8 63 --max-tokens 100000000 && \
nice -n 20 $PY -m analysis.look_write.fit_arms attn_s1 attn_s2 attn_s3 attn_s4 attn_s0 --band 8 63
echo "== calibration: BGU pass on one-layer models (20M tokens, CPU) $(date)"
nice -n 20 $PY -m analysis.look_write_deep.pass_bgu attn_s1 attn_s2 attn_s3 attn_s4 --tokens 20000000 --device cpu
for p in ltotal total; do
  $PY -m analysis.look_write_deep.fit_arms_deep attn_s1 attn_s2 attn_s3 attn_s4 --primary $p
done
echo "== done $(date)"
