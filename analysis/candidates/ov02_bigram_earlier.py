"""ov02 candidate: LOCAL-0 and LOCAL-2 also vote, from each EARLIER token they read, for that token's usual successors
(its corpus bigram PMI_1), as if it were the current token.

Mechanism (open-track insight, analysis/investigations/ov02/NOTES.md): a head's value/output map is the same for the
current token and for earlier tokens (it does not depend on distance). Head 0 puts about half its attention on the
current token, where its job is the bigram, so its map carries 'what follows s' and it writes the same thing when it
reads s one or more tokens back. Row correlation of the heads' OV maps with corpus tables: PMI_1 0.41 (head 0) / 0.36
(head 2), higher than with the distance-matched skip-grams PMI_2 0.37 / 0.29.

Directions: for h = 0, 2: sum_{j<i} a_h(i, j) PMI_1(t_j, v) with the program's attention rules (2 strengths).
Counting rule: PMI_1 is the program's own bigram PMI table (corpus.py); nothing new is counted.
"""
import numpy as np

NAME = "ov02_bigram_earlier"
HEADS = [0, 2]


def features(P, tokens):
    from program import head0, head2
    n = len(tokens)
    pmi1 = np.asarray(P.STATS.pmi[1], np.float64)[tokens]
    out = {}
    for h, mod in ((0, head0), (2, head2)):
        a = mod.attention(tokens, P.STATS) * (1.0 - np.eye(n))
        out[f"h{h}.bigram"] = a @ pmi1
    return out
