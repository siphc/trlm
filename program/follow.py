"""FOLLOW: what usually comes next after the current token, ignoring everything earlier.

Score for next token v after current token t (tables in tables/follow.py):

    WORD[v]  +  CLASS_FOLLOW[row class of t][column class of v]  +  AFTER exceptions (t, v)

tables/follow.py names the class systems of CLASS_FOLLOW (ROWS, COLUMNS; see categories.py): the CONTEXT class
of the current token picks the row and the TARGET class of the next token picks the column.

CLAIM: (filled in after measurement)
"""
from .tables import follow as T
from .tables_lib import class_by_class, expand_classes, pairs_matrix, per_token

CLASS_PART = expand_classes(class_by_class(T.CLASS_FOLLOW, T.ROWS, T.COLUMNS), T.ROWS, T.COLUMNS)
FOLLOW = per_token(T.WORD)[None, :] + CLASS_PART + pairs_matrix(T.AFTER)


def votes(tokens):
    """Scores for every next token at each position, shape (n, VOCAB_SIZE)."""
    return FOLLOW[tokens]
