"""HEAD 2 of the model's single attention layer: reads the last few words; opens quotes after 'said,'.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head2.py):
  Role     Head 2 reads the last few tokens: 25% of its attention on the previous token, 33% on 2 to
           8 back, 6% further (mean distance 3.6), almost nothing (0.03%) more than 32 back. It
           abstains on 36%. It turns FOLLOW's one-token guess into a phrase-aware one
           ('He closed his' -> ' eyes').
  Dialogue After a speech verb and a comma ('Tim said,'), the opening quote is the model's top token at
           94% of 247 held-out positions; the program gives it 0.86 with head 2 and 0.13 without. Right after a speech verb its
           removal cost is 3.9x its cost elsewhere.
  Removed  The most important head: replacing its votes by their average raises the KL by 0.27 nats;
           top-1 agreement with the model falls from 69% to 56%.
  Model    Ablating the earlier-token part of the model's head 2 costs 0.49 nats; the program head
           carries 55% of that, the closest match of the four, with the same speech-verb focus (model:
           3.2x). After ' and' the program leans on this head more than the model leans on its head 2
           (0.51 vs 0.22 nats).
"""
from .attention import Head
from .tables import head2 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
