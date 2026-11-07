#!/bin/bash
# Build and score the L1 account for every seed of one arm: P1 (eval.py) and P3 (scrub.py).
#   scripts/ladder/l1_accounts.sh ln
cd "$(dirname "$0")/../.."
mkdir -p analysis/out/ladder/l1
for S in 0 1 2 3 4; do
  .venv/bin/python -m analysis.ladder.l1.build_account --arm $1 --seed $S >/dev/null
  .venv/bin/python -m analysis.ladder.eval --rung L1 --arm $1 --seed $S --account ladder/L1_$1/account_s$S.py > analysis/out/ladder/l1/p1_$1_s$S.json
  .venv/bin/python -m analysis.ladder.scrub --rung L1 --arm $1 --seed $S --account ladder/L1_$1/account_s$S.py > analysis/out/ladder/l1/p3_$1_s$S.json
done
.venv/bin/python - $1 <<'PY'
import json, sys
arm = sys.argv[1]
for s in range(5):
    p1 = json.load(open(f"analysis/out/ladder/l1/p1_{arm}_s{s}.json")); p3 = json.load(open(f"analysis/out/ladder/l1/p3_{arm}_s{s}.json"))
    c = {v["site"]: v["faithfulness"] for v in p3["claims"].values()}
    print(f"{arm} s{s}: KL {p1['kl']:.4f} floor {p1['kl_floor']:.4f} P1 {p1['P1']} | direct {c['direct']:.4f} head {c['head0']:.4f} joint {p3['joint']['faithfulness']:.4f} P3 {p3['P3']}")
PY
