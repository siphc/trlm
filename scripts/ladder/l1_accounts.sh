#!/bin/bash
# Build and score the L1 account (analysis/ladder/l1/build_account.py) for every seed of one arm:
# P1 (eval.py), P3 (scrub.py) per seed, P4 (behaviour.py) over the five seeds and the untrained model.
#   scripts/ladder/l1_accounts.sh ln          (P2: add P2=1 to also run length.py, slower)
cd "$(dirname "$0")/../.."
OUT=analysis/out/ladder/l1/v2; mkdir -p $OUT
for S in 0 1 2 3 4; do
  .venv/bin/python -m analysis.ladder.l1.build_account --arm $1 --seed $S >/dev/null
  .venv/bin/python -m analysis.ladder.eval --rung L1 --arm $1 --seed $S --account ladder/L1_$1/account_s$S.py > $OUT/p1_$1_s$S.json
  .venv/bin/python -m analysis.ladder.scrub --rung L1 --arm $1 --seed $S --account ladder/L1_$1/account_s$S.py > $OUT/p3_$1_s$S.json
  [ -n "$P2" ] && .venv/bin/python -m analysis.ladder.length --rung L1 --arm $1 --seed $S --account ladder/L1_$1/account_s$S.py > $OUT/p2_$1_s$S.json
done
.venv/bin/python -m analysis.ladder.behaviour --rung L1 --arm $1 --account "ladder/L1_$1/account_s{seed}.py" > $OUT/p4_$1.json
.venv/bin/python - $1 $OUT <<'PY'
import json, os, sys
arm, out = sys.argv[1], sys.argv[2]
for s in range(5):
    p1 = json.load(open(f"{out}/p1_{arm}_s{s}.json")); p3 = json.load(open(f"{out}/p3_{arm}_s{s}.json"))
    c = {v["site"]: v["faithfulness"] for v in p3["claims"].values()}
    line = (f"{arm} s{s}: KL {p1['kl']:.4f} floor {p1['kl_floor']:.4f} P1 {p1['P1']} | direct {c['direct']:.4f} "
            f"head {c['head0']:.4f} joint {p3['joint']['faithfulness']:.4f} P3 {p3['P3']}")
    if os.path.exists(f"{out}/p2_{arm}_s{s}.json"):
        p2 = json.load(open(f"{out}/p2_{arm}_s{s}.json"))
        line += f" | bits {p2['account']['bits']} ref {p2['text_reference']['total_bits']} P2 {p2['P2']}"
    print(line)
p4 = json.load(open(f"{out}/p4_{arm}.json"))
for name, c in p4["claims"].items():
    for t in c["tests"]:
        v = t["values"]
        print(f"P4 {name}: {t['statement']} ({t['op']} {t['threshold']}): "
              + " ".join(f"{k} {v[k]:.3f}" for k in v) + f" -> {'pass' if t['pass'] else 'FAIL'}")
print("P4", p4["P4"])
PY
