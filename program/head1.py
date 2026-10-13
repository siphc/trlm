"""HEAD 1 of the model's single attention layer: recalls words from far back and votes to repeat them.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head1.py):
  Role     Head 1 looks far back: mean distance 34 tokens, 61% of its attention more than 8 tokens
           back, almost none on the previous token (0.7%), nothing (under 0.1%) more than 128 back.
           Names get 3.4x their share of its attention. It abstains on 29%. A token it attends to
           votes for itself, typically 17 nats more than for other words of its class.
  Recall   At recall positions (the model's top token appeared earlier in the window, but FOLLOW
           alone gives it under 5%; one position in five) head 1 raises the log-probability of that
           token by 1.03 nats, against 0.14 elsewhere (7.4x).
  Removed  Replacing its votes by their average raises the KL by 0.18 nats; at recall positions
           top-1 agreement with the model falls from 59% to 36%.
  Model    Ablating the earlier-token part of the model's head 1 costs 0.42 nats; the program head
           carries 42% of that, with the same focus on recall positions (model: 1.53 vs 0.29 nats,
           5.3x). Unlike this head, the model head rarely looks at the current token (4%), and it
           looks less far back (mean distance 27).
"""
from .attention import Head
from .tables import head1 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
