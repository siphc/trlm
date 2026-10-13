"""Replacement program for a one-block, attention-only TinyStories language model.

The whole algorithm, top-down (each step is one module):

  1. follow.py   Each token votes for what usually comes right after it, ignoring earlier context.
  2. head0-3.py  Four attention heads each look back over the text, pick out earlier tokens they
                 care about, and vote for next tokens related to those tokens.
  3. The votes are added and turned into next-token probabilities with a softmax.

Tables live in tables/; vocab.py and categories.py define the tokens and their word classes.

CLAIM (measured on 200 held-out text windows by analysis/claims_measure.py, checked by
tests/test_program.py; tests/measurements.py defines every quantity used in the CLAIMs):
  - The scores are exactly FOLLOW plus the four heads' votes; step 3 adds nothing else.
  - The program's next-token distribution is 0.47 nats (KL) from the model's, and its top token is
    the model's top token 69% of the time.
  - FOLLOW alone is 1.40 nats from the model (top-1 34%). The four heads remove 0.93 nats of that,
    two thirds.
"""
import numpy as np

from . import follow, head0, head1, head2, head3

HEADS = [head0, head1, head2, head3]


def predict(tokens):
    """Log-probabilities, shape (len(tokens), 2000). Row i is the next-token distribution after
    tokens[: i + 1]."""
    tokens = np.asarray(tokens, dtype=int)
    score = follow.votes(tokens)
    for head in HEADS:
        score = score + head.votes(tokens)
    return log_softmax(score)


def log_softmax(score):
    score = score - score.max(axis=1, keepdims=True)
    return score - np.log(np.exp(score).sum(axis=1, keepdims=True))
