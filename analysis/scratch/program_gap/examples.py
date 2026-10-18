import numpy as np
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.program_gap.bottleneck import kl_rows
from program.vocab import TOKENS
P = program()
W = train_windows(4, seed=EVAL_SEED)
lp = model_logprobs(W)
for wi in range(2):
    t = np.asarray(W[wi, :256], np.int64)
    lq = P.predict(t)
    kl = kl_rows(lp[wi], lq)
    order = np.argsort(-kl)[:14]
    for i in sorted(order):
        ctx = "".join(TOKENS[x] for x in t[max(0, i - 12): i + 1]).replace("\n", "/")
        m = np.argsort(-lp[wi, i])[:6]; q = np.argsort(-lq[i])[:6]
        fm = " ".join(f"{TOKENS[x]!r}:{np.exp(lp[wi,i,x]):.2f}" for x in m)
        fq = " ".join(f"{TOKENS[x]!r}:{np.exp(lq[i,x]):.2f}" for x in q)
        print(f"[{i}] KL {kl[i]:.2f} ...{ctx!r} -> next {TOKENS[t[i+1]] if i < 255 else '?'!r}\n   model: {fm}\n   prog:  {fq}")
