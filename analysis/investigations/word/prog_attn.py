"""Program attention rules (head0, head2, head1) on the same roles as attn_dump.py, at word-internal queries."""
import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, program
from program import head0, head1, head2

P = program()
PIECE = WS.piece_flags()
W = train_windows(32, seed=EVAL_SEED)
roles = ["self", "word start", "earlier pieces", "token before word", "prev 1-3", "rest"]
acc = {}
for w in W:
    t = np.asarray(w[:256], dtype=np.int64)
    cl, Ls = classes(P, t, PIECE)
    A = {h: m.attention(t, P.STATS) for h, m in (("h0", head0), ("h2", head2), ("h1", head1))}
    for i in range(1, len(t)):
        c = cl[i] if not cl[i].startswith("inside") else ("inside L=2" if Ls[i] == 2 else "inside L>=3")
        L = Ls[i]
        j0 = i - L + 1
        role = np.full(i + 1, 5)
        role[max(i - 3, 0): i] = 4
        if L >= 2:
            role[j0 + 1: i] = 2
            role[j0] = 1
            if j0 >= 1:
                role[j0 - 1] = 3
        role[i] = 0
        r = acc.setdefault(c, {h: np.zeros(7) for h in A})
        for h, a in A.items():
            for k in range(6):
                r[h][k] += a[i, : i + 1][role == k].sum()
            r[h][6] += 1
for c in ["inside L=2", "inside L>=3", "onset", "after multi-token word", "other"]:
    print(c)
    for h, r in acc[c].items():
        print(f"   prog {h}", "  ".join(f"{roles[k]} {r[k] / r[6]:.3f}" for k in range(6)))
