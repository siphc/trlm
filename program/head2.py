"""HEAD 2 of the model's single attention layer: completes phrases from the last few words; opens quotes after 'said,'.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head2.py):
  Role     Head 2 reads the last few tokens: 17% of its attention on the previous token, 37% on 2 to
           8 back, 10% further (mean distance 5.0), nothing (under 0.1%) more than 40 back. It
           abstains on 36%. It completes phrases that the current token alone does not settle: after
           'The moral of the' the program gives ' story' 0.97, and 0.03 without this head.
  Dialogue After a speech verb and a comma ('Tim said,'), the opening quote is the model's top token at
           94% of 247 held-out positions; the program gives it 0.87 with head 2 and 0.26 without.
           Right after a speech verb its removal cost is 3.5x its cost elsewhere.
  Removed  Replacing its votes by their average raises the KL by 0.19 nats; top-1 agreement with the
           model falls from 69% to 60%.
  Model    Ablating the earlier-token part of the model's head 2 costs 0.49 nats; the program head
           carries 39% of that, with the same speech-verb focus (model: 3.2x; without its head 2 the
           model's ' story' falls from 0.91 to under 0.01). Some of the model head's work sits in other
           program heads. Inside split words it matters more than any other model head (0.94 nats),
           this head only 0.27. The model needs its head 2 for 'm' after 'She had a proble' and for
           ' eyes' after 'He closed his'; the program needs head 0 and head 3 for them instead.
"""
from .attention import Head
from .tables import head2 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
