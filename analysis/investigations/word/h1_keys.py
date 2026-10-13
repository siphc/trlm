"""Head 1 vs RECALL's rule: attention mass on earlier keys by key kind (onset / piece / other word), all queries."""
import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.decomp import parts
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model
from program import head1

P = program()
s = P.STATS
PIECE = WS.piece_flags()
model = load_model("attn_s0")
W = train_windows(32, seed=EVAL_SEED)
acc = {}
with torch.no_grad():
    for w in W:
        t = np.asarray(w[:256])
        a, _, _, _ = parts(model, torch.as_tensor(t[None]))
        A = a[0, 1].double().numpy()
        R = head1.attention(t, s)
        n = len(t)
        kind = np.where(PIECE[t], "piece", np.where((s.cont[t] > 0.5), "onset", "word/other"))
        cl, _ = classes(P, t, PIECE)
        cl = np.array(["inside" if c.startswith("inside") else c for c in cl], dtype=object)
        for i in range(1, n):
            for k in ("onset", "piece", "word/other"):
                m = kind[:i] == k
                r = acc.setdefault((cl[i], k), [0, 0.0, 0.0])
                r[0] += 1; r[1] += A[i, :i][m].sum(); r[2] += R[i, :i][m].sum()
for (c, k), r in sorted(acc.items()):
    if c in ("pos0",):
        continue
    print(f"query {c:24s} key {k:10s}: head {r[1] / r[0]:.3f}  rule {r[2] / r[0]:.3f}")
