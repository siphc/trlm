"""Where does the program's KL sit at word positions? Split KL(model || program) into
  (a) the two-way part: KL between Bernoulli(P(next is a piece)) of model and program
  (b) within pieces and within non-pieces (conditional KLs, weighted by the model's mass).
KL = (a) + P_m(piece) KL(m|piece || q|piece) + P_m(non) KL(m|non || q|non). Also the same for the corpus bigram.
"""
import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

P = program()
PIECE = WS.piece_flags()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
acc = {}


def split(p, q):
    pm, qm = np.exp(p), np.exp(q)
    a_m, a_q = pm[:, PIECE].sum(1), qm[:, PIECE].sum(1)
    two = a_m * np.log(a_m / a_q) + (1 - a_m) * np.log((1 - a_m) / (1 - a_q))
    out = [two]
    for mask, am, aq in ((PIECE, a_m, a_q), (~PIECE, 1 - a_m, 1 - a_q)):
        pc = pm[:, mask] / am[:, None]
        qc = qm[:, mask] / aq[:, None]
        out.append(am * (pc * (np.log(pc + 1e-300) - np.log(qc))).sum(1))
    return np.stack(out, 1)


for w, p in zip(W, lp):
    t = np.asarray(w[:256])
    q = P.predict(t)
    cl, Ls = classes(P, t, PIECE)
    sp = split(p, q)
    for i in range(len(t)):
        c = cl[i]
        acc.setdefault(c, []).append(sp[i])
print("class: n, KL total, two-way (piece vs not), within pieces, within non-pieces")
tot = sum(np.sum(v) for v in acc.values())
for c, v in sorted(acc.items()):
    v = np.array(v)
    print(f"  {c:25s} {len(v):5d}  {v.sum(1).mean():.3f}  {v[:, 0].mean():.3f}  {v[:, 1].mean():.3f}  {v[:, 2].mean():.3f}"
          f"   share of all KL: {v.sum() / tot:.3f} (two-way {v[:, 0].sum() / tot:.3f}, pieces {v[:, 1].sum() / tot:.3f}, non {v[:, 2].sum() / tot:.3f})")
