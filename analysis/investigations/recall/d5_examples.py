"""D5: example rows: head 1's top keys vs RECALL's top keys for frequent query tokens."""
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402
from program.vocab import TOKENS, ID  # noqa: E402

P = program()
net = lib.net()
W = train_windows(4, seed=2000)
want = {ID[x]: 0 for x in [".", " the", "\n", " was", " a", " her", " little", ",", " to", " said", " and"]}
for w in W:
    t = np.asarray(w[:256], np.int64)
    A = net.forward(t)["patterns"][1]
    R = head1.attention(t, P.STATS)
    for i in range(40, 256):
        if t[i] in want and want[t[i]] < 2:
            want[t[i]] += 1
            ctx = "".join(TOKENS[x] for x in t[max(0, i - 12): i + 1]).replace("\n", "\\n")
            nxt = TOKENS[t[i + 1]] if i + 1 < 256 else "?"
            fm = lambda M: ", ".join(f"{TOKENS[t[j]]!r}@{i - j}:{M[i, j]:.2f}" for j in np.argsort(-M[i])[:7])
            print(f"q={TOKENS[t[i]]!r} i={i} ...{ctx!r} -> {nxt!r}\n  model: {fm(A)}\n  rule : {fm(R)}")
