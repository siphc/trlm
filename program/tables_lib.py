"""Turns the readable tables in program/tables/ into arrays. No numbers live here.

Every table in program/tables/ is written in one of three shapes:
  per-token values, grouped by word class:   TABLE["NAMES"] = {" Lily": 1.2, " Tom": 0.9}
  class-by-class values, one row per class:  TABLE["POSSESSIVES"] = {"BELONGINGS": 2.1, "PLACES": 1.4}
  token-pair exceptions, grouped by class:   pair(TABLE, " upon", A_AND_AN={" a": 3.0})
Values are in nats (natural-log units) and are added to scores before the softmax. A missing entry
means 0.

Classes come from one of the class systems in categories.py ("ATTENTION", "CONTEXT", "TARGET"). The rows and
the columns of a class-by-class table may use different systems; each table file names the systems it uses.
"""
import numpy as np

from .categories import CLASS_SYSTEMS
from .vocab import ID, VOCAB_SIZE

# For each class system: class name -> class number.
CLASS_INDEX = {system: {name: k for k, name in enumerate(classes)} for system, classes in CLASS_SYSTEMS.items()}


def _token_classes(system):
    out = np.zeros(VOCAB_SIZE, dtype=int)
    for name, tokens in CLASS_SYSTEMS[system].items():
        out[[ID[t] for t in tokens]] = CLASS_INDEX[system][name]
    return out


# For each class system: the class number of every token.
TOKEN_CLASS = {system: _token_classes(system) for system in CLASS_SYSTEMS}


def per_token(table, default=0.0):
    out = np.full(VOCAB_SIZE, float(default))
    for group in table.values():
        for text, value in group.items():
            out[ID[text]] = value
    return out


def class_by_class(table, rows, columns):
    """A class-by-class table as a matrix; rows and columns name class systems."""
    out = np.zeros((len(CLASS_INDEX[rows]), len(CLASS_INDEX[columns])))
    for row, group in table.items():
        for col, value in group.items():
            out[CLASS_INDEX[rows][row], CLASS_INDEX[columns][col]] = value
    return out


def per_class(table, system):
    out = np.zeros(len(CLASS_INDEX[system]))
    for name, value in table.items():
        out[CLASS_INDEX[system][name]] = value
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


def expand_classes(matrix, rows, columns):
    """A class-by-class matrix as a token-by-token matrix (row token's class in system `rows`, column token's
    class in system `columns`)."""
    return matrix[TOKEN_CLASS[rows]][:, TOKEN_CLASS[columns]]
