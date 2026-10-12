"""HEAD 1 of the model's single attention layer.

CLAIM: (filled in after measurement)
"""
from .attention import Head
from .tables import head1 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
