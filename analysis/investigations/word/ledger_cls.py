"""KL(model || program) by word-position class with each mechanism switched off (no refit), 64 TRAIN windows."""
import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M

P = program()
PIECE = WS.piece_flags()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
names = [m.NAME for m in P.MECHANISMS]
acc = {}
for w, p in zip(W, lp):
    t = np.asarray(w[:256])
    cl, Ls = classes(P, t, PIECE)
    cl = np.array(["inside" if c.startswith("inside") else c for c in cl], dtype=object)
    ks = {"all": M.kl(p, P.predict(t))}
    for m in names:
        ks[m] = M.kl(p, P.predict(t, off=(m,)))
    for c in set(cl):
        r = acc.setdefault(c, {k: [] for k in ks})
        for k, v in ks.items():
            r[k].append(v[cl == c])
for c, r in sorted(acc.items()):
    base = np.concatenate(r["all"])
    print(f"{c:25s} n {len(base):5d} KL {base.mean():.3f}  " + "  ".join(f"-{m} {np.concatenate(r[m]).mean() - base.mean():+.3f}" for m in names))
