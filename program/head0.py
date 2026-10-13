"""HEAD 0 of the model's single attention layer: passes on the last few tokens and finishes split words.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head0.py):
  Role     Head 0 abstains on 52% of its attention (the current token: no vote, see attention.py).
           The rest comes from close by: 18% on the previous token, 20% on 2 to 8 tokens back, 10%
           further (mean distance 5.4 tokens), and nothing (under 0.1%) more than 48 tokens back.
  Words    It matters most inside words that the tokenizer splits into pieces (' pro' 'ble' -> 'm',
           as in 'problem'): its removal cost there is 3.0x its cost elsewhere (0.35 vs 0.12 nats).
           In absolute terms head 2 matters slightly more inside words (0.39 nats).
  Removed  Replacing its votes by their average raises the KL by 0.15 nats; top-1 agreement with the
           model falls from 69% to 63%.
  Model    Ablating the earlier-token part of the model's head 0 costs 0.46 nats; the program head
           carries a third of that, with the same inside-word focus (model: 2.0x). The model head's
           49% attention on the current token matches this head's 52% abstention; in the program that
           part of the job is done by FOLLOW.
"""
from .attention import Head
from .tables import head0 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
