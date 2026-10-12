"""FOLLOW: what usually comes next after the current token, ignoring everything earlier.

Score for next token v after current token t (tables in tables/follow.py):

    WORD[v]  +  CLASS_FOLLOW[class of t][class of v]  +  AFTER exceptions (t, v)

CLAIM: (filled in after measurement)
"""
from .tables import follow as T
from .tables_lib import class_by_class, expand_classes, pairs_matrix, per_token

FOLLOW = per_token(T.WORD)[None, :] + expand_classes(class_by_class(T.CLASS_FOLLOW)) + pairs_matrix(T.AFTER)


def votes(tokens):
    """Scores for every next token at each position, shape (n, VOCAB_SIZE)."""
    return FOLLOW[tokens]
