import numpy as np, sys
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, program
from analysis.program_gap.budget import program_parts
from program.vocab import TOKENS, ID
P = program(); s = P.STATS
W = train_windows(64, seed=EVAL_SEED)
def show(t, i, cands):
    t = np.asarray(t, np.int64)
    pp = program_parts(P, t)
    ctx = "".join(TOKENS[x] for x in t[max(0, i-8): i+1])
    print(f"== pos {i} ...{ctx!r}")
    for c in cands:
        v = ID[c]
        print(f"  {c!r:10s} " + " ".join(f"{k}:{pp[k][i, v]:+.2f}" for k in pp) + f" | total {sum(pp[k][i, v] for k in pp):+.2f}")
    print("  log P1:", {c: round(float(s.log_follow[t[i], ID[c]]), 2) for c in cands})
toks = np.asarray(W[:, :257], np.int64)
done = 0
for w in range(64):
    for i in range(1, 256):
        if TOKENS[toks[w, i]] == "Once" and toks[w, i-1] == 0 and done < 1:
            show(toks[w, :256], i, [" upon", " there", " a"]); done += 1
        if TOKENS[toks[w, i]] == " and" and TOKENS[toks[w, i-1]] == " Jane":
            show(toks[w, :256], i, [" Jane", " her", " Tom", " she"])
for w in range(64):
    if TOKENS[toks[w, 0]] in ("air", " tree"):
        show(toks[w, :256], 0, [TOKENS[toks[w, 0]], ".", " and", " bran"])
