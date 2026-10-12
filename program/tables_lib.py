"""Turns the readable tables in program/tables/ into arrays. No numbers live here.

Every table in program/tables/ is written in one of three shapes:
  per-token values, grouped by word class:   TABLE["NAMES"] = {" Lily": 1.2, " Tom": 0.9}
  class-by-class values, one row per class:  TABLE["DETERMINERS"] = {"CREATURES": 2.1, "PLACES": 1.4}
  token-pair exceptions, grouped by class:   pair(TABLE, " upon", CREATURES={" a": 3.0})
Values are in nats (natural-log units) and are added to scores before the softmax. A missing entry
means 0.
"""
import numpy as np

from .categories import CLASS_NAMES, CLASS_OF
from .vocab import ID, VOCAB_SIZE

CLASS_INDEX = {name: k for k, name in enumerate(CLASS_NAMES)}
TOKEN_CLASS = np.array([CLASS_INDEX[CLASS_OF[i]] for i in range(VOCAB_SIZE)])


def per_token(table, default=0.0):
    out = np.full(VOCAB_SIZE, float(default))
    for group in table.values():
        for text, value in group.items():
            out[ID[text]] = value
    return out


def class_by_class(table):
    out = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
    for row, group in table.items():
        for col, value in group.items():
            out[CLASS_INDEX[row], CLASS_INDEX[col]] = value
    return out


def per_class(table):
    out = np.zeros(len(CLASS_NAMES))
    for name, value in table.items():
        out[CLASS_INDEX[name]] = value
    return out


def pair(table, first, **groups):
    """Record exceptions for token `first`: groups map a class name to {second token: value}."""
    row = table.setdefault(first, {})
    for group in groups.values():
        row.update(group)


def pairs_matrix(table):
    out = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
    for first, row in table.items():
        for second, value in row.items():
            out[ID[first], ID[second]] = value
    return out


def expand_classes(matrix):
    """A class-by-class matrix as a token-by-token matrix."""
    return matrix[TOKEN_CLASS][:, TOKEN_CLASS]
