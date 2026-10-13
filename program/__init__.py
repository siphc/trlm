"""Mechanism program for attn_s0, a one-block, attention-only TinyStories language model.

The program explains the model's next-token predictions as a few readable mechanisms. Every token-level number
comes from statistics of the training stories, counted when the program starts (setup); only a few dozen named
scalars were fitted to the model, and each attention rule was fitted to the matching model head's own attention.

The whole algorithm, top-down (one module per mechanism):

  setup(corpus)   corpus.py  count the statistics of the training stories
  predict(tokens) for every position i, add the votes of
                  follow.py  FOLLOW   what usually comes right after the current token (corpus bigram)
                  head0.py   LOCAL-0  the last few tokens (steep recency) vote for what follows them at that
                                      distance (corpus skip-grams)                         [model head 0]
                  head2.py   LOCAL-2  informative recent words, skipping word pieces, vote the same way [head 2]
                  head1.py   RECALL   earlier story words that could come next are recalled and vote for
                                      themselves, their topic and what follows them        [model head 1]
                  phrase.py  PHRASE   continue the last two or three tokens as a known phrase (corpus 3/4-grams)
                  and turn the summed votes into probabilities with a softmax.

Model head 3 is not modelled in this version (logs/mechanism_cards.md: no vote rule tried so far explains it).

CLAIM (tests/test_program.py): on held-out text the five mechanisms together are within 0.62 nats of the model,
and each is needed: switching one off raises KL by at least 0.05 nats (FOLLOW by at least 1.0).
"""
import numpy as np

from . import corpus as corpus_counts
from . import follow, head0, head1, head2, phrase

STATS = None
MECHANISMS = [follow, head0, head2, head1, phrase]


def setup(corpus):
    """Count the training-story statistics once (the harness passes the TRAIN token stream)."""
    global STATS
    STATS = corpus_counts.build(corpus)


def predict(tokens, off=()):
    """Log-probabilities, shape (len(tokens), 2000); row i is the next-token distribution after tokens[: i + 1].
    `off` names mechanisms to switch off (for removal tests)."""
    if STATS is None:
        raise RuntimeError("call setup(corpus) first")
    tokens = np.asarray(tokens, dtype=np.int64)
    score = np.zeros((len(tokens), 2000))
    for mechanism in MECHANISMS:
        if mechanism.NAME not in off:
            score += mechanism.votes(tokens, STATS)
    return log_softmax(score)


def log_softmax(score):
    score = score - score.max(axis=1, keepdims=True)
    return score - np.log(np.exp(score).sum(axis=1, keepdims=True))
