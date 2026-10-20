#!/bin/bash
# Run ON THE INSTANCE once, in ~/trlm: Python 3.12 environment with the laptop's package versions, data check, corpus
# statistics, study A's usefulness tables (checked against the committed prediction), sanity tests, GPU step times,
# and the idle guard.  Run: bash scripts/aws/setup_remote.sh 2>&1 | tee -a logs/runs/aws_setup.log
set -e
cd ~/trlm
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.14.1 numpy==2.5.4 pytest==9.1.1 PyYAML==6.0.3 tokenizers==0.23.3 \
    huggingface_hub==2.2.0 pyarrow==26.0.0
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
.venv/bin/python scripts/aws/manifest.py check
.venv/bin/python -c "from analysis.mech_eval import stats; stats()"            # program corpus tables (~2-3 min)
cp analysis/out/fixed_votes/usefulness/profile.json /tmp/profile_laptop.json
OMP_NUM_THREADS=4 .venv/bin/python -m analysis.fixed_votes.usefulness
.venv/bin/python - <<'PY'
import json
a, b = (json.load(open(p))["mean_usefulness"] for p in ("/tmp/profile_laptop.json", "analysis/out/fixed_votes/usefulness/profile.json"))
err = max(abs(a[t][d] - b[t][d]) for t in a for d in a[t])
print("usefulness profile max abs difference vs laptop:", err)
assert err < 1e-6
PY
OMP_NUM_THREADS=8 .venv/bin/python -m pytest -q src/train/test_sanity.py src/train/test_sanity_fixed_votes.py src/train/test_sanity_law_pressure.py
.venv/bin/python -m analysis.fixed_votes.rules r2 --device cuda
.venv/bin/python -m analysis.law_pressure.rules r2 --device cuda || true     # timing only; R2's value was fixed on the laptop
nohup sh scripts/aws/idle_guard.sh > /dev/null 2>&1 &
echo "setup done $(date)"
