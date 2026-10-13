"""FOLLOW: what usually comes next after the current token, ignoring everything earlier.

Score for next token v after current token t (tables in tables/follow.py):

    WORD[v]  +  CLASS_FOLLOW[row class of t][column class of v]  +  AFTER exceptions (t, v)

tables/follow.py names the class systems of CLASS_FOLLOW (ROWS, COLUMNS; see categories.py): the CONTEXT class
of the current token picks the row and the TARGET class of the next token picks the column.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_follow.py):
  Role     FOLLOW is the program's guess from the current token alone, a bigram table. On its own it
           is 1.39 nats from the model (KL) and picks the model's top token 34% of the time.
  Phrases  Fixed phrases need nothing else: after ' upon' it gives ' a' 0.92, and ' a' is the model's
           top token at all 140 held-out ' upon's (likewise ' wanted' -> ' to': 0.87, model 0.87).
           Not every phrase: after 'Once' FOLLOW alone gives ' upon' 0.38 against the model's 0.88;
           the heads supply the rest, mostly head 2 (without head 2: 0.51). KL there is 0.66 for
           FOLLOW alone, 0.03 for the whole program.
  Removed  Its 31k token-pair exceptions (AFTER) are the largest single part of the program:
           replacing them by their average raises the KL by 0.89 nats, more than doing the same to its
           class table CLASS_FOLLOW (0.75) and over four times the cost of any one head (at most 0.21).
  Model    The model has no separate FOLLOW. The current token's own vote travels through model
           heads 0 and 2 attending to the current token (49% and 34% of their attention) and through
           the direct embedding path.
"""
from .tables import follow as T
from .tables_lib import class_by_class, expand_classes, pairs_matrix, per_token

CLASS_PART = expand_classes(class_by_class(T.CLASS_FOLLOW, T.ROWS, T.COLUMNS), T.ROWS, T.COLUMNS)
FOLLOW = per_token(T.WORD)[None, :] + CLASS_PART + pairs_matrix(T.AFTER)


def votes(tokens):
    """Scores for every next token at each position, shape (n, VOCAB_SIZE)."""
    return FOLLOW[tokens]
