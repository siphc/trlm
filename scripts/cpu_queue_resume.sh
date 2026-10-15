#!/bin/sh
# CPU analysis of the deep study, resumable (finished passes and arm files are skipped). One process at a time,
# lowest priority, 2 threads.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2
PY="nice -n 20 .venv/bin/python"
echo "== resume: one-layer code-check passes (20M tokens; finished ones are skipped) $(date)"
$PY -m analysis.look_write_deep.pass_bgu attn_s1 attn_s2 attn_s3 attn_s4 --tokens 20000000 --device cpu
for p in ltotal total; do
  for m in attn_s1 attn_s2 attn_s3 attn_s4; do
    [ -f analysis/out/look_write_deep/$m/arms_$p.json ] || $PY -m analysis.look_write_deep.fit_arms_deep $m --primary $p
  done
done
echo "== D: echo trajectory $(date)"
$PY -m analysis.look_write_deep.echo_traj --tokens 5000000 --device cpu
echo "== done $(date)"
