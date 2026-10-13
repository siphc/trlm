"""Replacement program for a one-block, attention-only TinyStories language model.

The whole algorithm, top-down (each step is one module):

  1. follow.py   Each token votes for what usually comes right after it, ignoring earlier context.
  2. head0-3.py  Four attention heads each look back over the text, pick out earlier tokens they
                 care about, and vote for next tokens related to those tokens.
  3. confidence.py  The summed votes are scaled by how sharp predictions are after the current token.
  4. The scaled votes are turned into next-token probabilities with a softmax.

Tables live in tables/; vocab.py and categories.py define the tokens and their word classes.
"""
import numpy as np

from . import confidence, follow, head0, head1, head2, head3

HEADS = [head0, head1, head2, head3]


def predict(tokens):
    """Log-probabilities, shape (len(tokens), 2000). Row i is the next-token distribution after
    tokens[: i + 1]."""
    tokens = np.asarray(tokens, dtype=int)
    score = follow.votes(tokens)
    for head in HEADS:
        score = score + head.votes(tokens)
    return log_softmax(score * confidence.scale(tokens))


def log_softmax(score):
    score = score - score.max(axis=1, keepdims=True)
    return score - np.log(np.exp(score).sum(axis=1, keepdims=True))
