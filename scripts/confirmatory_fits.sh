#!/bin/sh
# Step 5.4 (second half): run ONLY after logs/prereg_deep_look_write.md is committed. Arm fits with the primary
# readout (CPU, two models at a time, as their passes finish), then the summary with the R1-R3 values, the code test
# on the held-out heads, and the capacity metrics on seeds 1-4.
cd "$(dirname "$0")/.." || exit 1
git log --oneline -- logs/prereg_deep_look_write.md | grep -q "FINAL" || { echo "pre-registration not committed as FINAL"; exit 1; }
OUT=analysis/out/look_write_deep
PRIMARY=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/rules.json'))['R1']['primary'])")
fit() {
  m=$1
  until [ -f $OUT/$m/bgu_meta.json ] && grep -q '"n_tokens": 2' $OUT/$m/bgu_meta.json; do sleep 60; done
  [ -f $OUT/$m/arms_$PRIMARY.json ] || OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python -m analysis.look_write_deep.fit_arms_deep $m --primary $PRIMARY
}
for pair in "attn2L_s1 attn2L_s2" "attn2L_s3 attn2L_s4" "attn4L_s1 attn4L_s2" "attn4L_s3 attn4L_s4"; do
  set -- $pair
  echo "== fits $1 $2 $(date)"
  fit $1 & fit $2 & wait
done
echo "== summary $(date)"
.venv/bin/python - <<PY
import json, subprocess, sys
r = json.load(open("$OUT/rules.json"))
subprocess.run([sys.executable, "-m", "analysis.look_write_deep.summarize_deep", "--primary", r["R1"]["primary"],
                "--theta", str(r["R2"]["theta"]), "--c1", str(r["R3"]["c1_median"]), "--c2", str(r["R3"]["c2_median"])], check=True)
PY
echo "== code test $(date)"
OMP_NUM_THREADS=4 .venv/bin/python -m analysis.look_write_deep.code_test attn_s1 attn_s2 attn_s3 attn_s4 attn2L_s1 attn2L_s2 attn2L_s3 attn2L_s4 attn4L_s1 attn4L_s2 attn4L_s3 attn4L_s4
echo "== capacity seeds 1-4 $(date)"
OMP_NUM_THREADS=4 .venv/bin/python -m analysis.look_write_deep.capacity attn2L_s1 attn2L_s2 attn2L_s3 attn2L_s4 attn4L_s1 attn4L_s2 attn4L_s3 attn4L_s4
echo "== confirmatory fits done $(date)"
