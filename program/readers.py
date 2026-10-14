"""How a local reader turns its attention into votes (shared by head0.py and head2.py).

A reader at position i looks at positions j <= i with attention a(i, j) (its own rule). Each token it reads votes
for what the training stories put at the next position, given that token and how far back it is:

    the current token (j = i):        STRENGTH_SELF * PMI_1(t_i, v)
    a token d = 1..7 back:            STRENGTH_NEAR * PMI_{d+1}(t_j, v)    (what comes d + 1 tokens after it)
    a token 8 or more back:           STRENGTH_FAR  * PMI_8(t_j, v)

PMI_k(s, v) = log P(v sits k tokens after s) - log P(v), from the training stories (corpus.py). The reader's
votes at i are the attention-weighted sum of these.
"""
import numpy as np

from .corpus import USE_MAX_D


def skipgram_votes(attention, tokens, stats, strength_self, strength_near, strength_far):
    n = len(tokens)
    out = strength_self * np.diagonal(attention)[:, None] * stats.pmi[1][tokens]
    for d in range(1, 8):
        if d < n:
            weight = np.diagonal(attention, offset=-d)                 # a(i, i - d) for i = d .. n-1
            out[d:] += strength_near * weight[:, None] * stats.pmi[d + 1][tokens[:-d]]
    if n > 8:
        dist = np.arange(n)[:, None] - np.arange(n)[None, :]
        far = np.where(dist >= 8, attention, 0.0)
        out += strength_far * (far @ stats.pmi[8][tokens])
    return out


def earlier_only(attention):
    """The attention a(i, j) on earlier tokens only (j < i); the current token's weight set to zero."""
    return attention * (1.0 - np.eye(attention.shape[0]))


def mass_on_tokens(attention, tokens):
    """(n, V): at each position i, the total attention on each vocabulary token v (sum of a(i, j) over t_j = v)."""
    n = len(tokens)
    out = np.zeros((n, 2000))
    np.add.at(out, (np.repeat(np.arange(n), n), np.tile(tokens, n)), attention.reshape(-1))
    return out


def vote_usefulness(tokens, stats, strengths):
    """Where a reader's own vote is useful. At (i, j) with d = i - j in 1..7: how well the vote of t_j (a token d
    back), cast with this reader's strengths, predicts what follows t_i beyond what t_i alone predicts
    (corpus.usefulness_tables). `strengths` maps vote kinds (skip, succ, alike, copy) to the reader's strengths.
    Returns the d = 1 part and the d = 2..7 part as two (n, n) matrices (0 elsewhere)."""
    n = len(tokens)
    tables = {"skip": stats.use_skip, "succ": stats.use_succ, "alike": stats.use_alike, "copy": stats.use_copy}
    near, far = np.zeros((n, n)), np.zeros((n, n))
    for d in range(1, min(USE_MAX_D, n - 1) + 1):
        i = np.arange(d, n)
        u = sum(w * tables[kind][d - 1][tokens[i - d], tokens[i]] for kind, w in strengths.items())
        (near if d == 1 else far)[i, i - d] = u
    return near, far


def softmax_rows(score):
    """Softmax over earlier positions j <= i (score is (n, n); entries above the diagonal are ignored)."""
    n = score.shape[0]
    allowed = np.tril(np.ones((n, n), dtype=bool))
    score = np.where(allowed, score, -np.inf)
    score = score - score.max(axis=1, keepdims=True)
    w = np.exp(score)
    return w / w.sum(axis=1, keepdims=True)
