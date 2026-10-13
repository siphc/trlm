"""CONFIDENCE: how sharp the prediction is after the current token.

All votes at a position are multiplied by exp(CONFIDENCE[current token]) before the softmax: a positive
value makes the program more certain after that token, a negative value less certain, 0 leaves the votes
as they are. It stands in for the model's final normalisation step, whose strength depends mostly on the
current token. Table: tables/confidence.py (log scale factors, grouped by word class).

CLAIM: (filled in after measurement)
"""
import numpy as np

from .tables import confidence as T
from .tables_lib import per_token

LOG_SCALE = per_token(T.CONFIDENCE)


def scale(tokens):
    """Factor for each position's votes, shape (n, 1)."""
    return np.exp(LOG_SCALE[tokens])[:, None]
