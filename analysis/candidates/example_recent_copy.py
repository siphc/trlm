"""Example candidate (template): LOCAL-0 also votes against repeating the tokens it reads.

Hypothesis: model head 0's earlier-token part vetoes repeating the last few tokens (analysis/cancellation.py found a
coefficient of -9.7 on this direction for head 0's effect, R^2 0.035 -> 0.060).
"""
import numpy as np

NAME = "recent_copy"
HEADS = [0]


def features(P, tokens):
    from program import head0

    n = len(tokens)
    a = head0.attention(tokens, P.STATS) * (1.0 - np.eye(n))     # earlier tokens only
    copy = np.zeros((n, 2000))
    np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(tokens, n)), a.reshape(-1))
    return {"copy": copy}
