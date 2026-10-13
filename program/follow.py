"""FOLLOW: what usually comes next after the current token, ignoring everything earlier.

Score for next token v after current token t (tables in tables/follow.py):

    WORD[v]  +  CLASS_FOLLOW[class of t][class of v]  +  AFTER exceptions (t, v)

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_follow.py):
  Role     FOLLOW is the program's guess from the current token alone, a bigram table. On its own it
           is 1.40 nats from the model (KL) and picks the model's top token 34% of the time.
  Phrases  Fixed phrases need nothing else: after ' upon' it gives ' a' 0.92, and ' a' is the model's
           top token at all 140 held-out ' upon's (likewise ' wanted' -> ' to': 0.87, model 0.87).
           Not every phrase: after 'Once' FOLLOW alone gives ' upon' 0.50 against the model's 0.88;
           the heads supply the rest (KL there 0.42 for FOLLOW alone, 0.03 for the whole program).
  Removed  Its 34k token-pair exceptions (AFTER) are the largest single part of the program:
           replacing them by their average raises the KL by 1.43 nats, more than removing all four
           heads does (0.93).
  Model    The model has no separate FOLLOW. The current token's own vote travels through model
           heads 0 and 2 attending to the current token (49% and 34% of their attention) and through
           the direct embedding path.
"""
from .tables import follow as T
from .tables_lib import class_by_class, expand_classes, pairs_matrix, per_token

FOLLOW = per_token(T.WORD)[None, :] + expand_classes(class_by_class(T.CLASS_FOLLOW)) + pairs_matrix(T.AFTER)


def votes(tokens):
    """Scores for every next token at each position, shape (n, VOCAB_SIZE)."""
    return FOLLOW[tokens]
