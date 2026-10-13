"""D9: what do function-word, sentence-end and word-end keys write (head 1 OV, exact, insight only) vs RECALL's
corpus votes for them (group copy + topic + next with the exact-attention strengths of D2)? Top targets by
probability-weighted deviation."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402
from program.vocab import TOKENS, ID  # noqa: E402

P = program()
s = P.STATS
OV = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/OV.npy").astype(np.float64)
d2 = json.load(open(f"{lib.OUT}/d2_votes.json"))["per group (target = exact contribution of that group)"]
pv = np.asarray(s.p_uni, np.float64)
for tk in [" she", " he", " her", " his", " they", " it", " I", " you", " the", " was", "\n", ".", '!"', ' "', ",",
           " and", "<|eos|>", " upon", "Once"]:
    i = ID[tk]
    g = head1.GROUP_NAMES[head1.GROUP[i]]
    c, tp, nx = d2[g]["copy,topic,next"]["all"]["beta"]
    pred = tp * s.topic[i] + nx * s.pmi[1][i]
    pred = np.asarray(pred, np.float64)
    pred[i] += c
    o = OV[i] - (OV[i] * pv).sum()
    pr = pred - (pred * pv).sum()
    top = np.argsort(-o)[:10]
    bot = np.argsort(o)[:6]
    print(f"{tk!r} [{g}] OV top: {[(TOKENS[j], round(o[j], 1)) for j in top]}")
    print(f"     OV bottom: {[(TOKENS[j], round(o[j], 1)) for j in bot]}")
    print(f"     program top: {[(TOKENS[j], round(pr[j], 1)) for j in np.argsort(-pr)[:8]]}")
