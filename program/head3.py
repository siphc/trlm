"""HEAD 3 of the model's single attention layer: carries a phrase across linking words, above all after ' to' and ' of'.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head3.py):
  Role     Head 3 reads the last few tokens: 16% of its attention on the previous token, 34% on 2 to
           8 back, 12% further (mean distance 5.5), nothing (under 0.1%) more than 40 back. It
           abstains on 37%.
  Linkers  It matters most right after linking words (' and', ',', ' to', ' of'): its removal cost
           there is 2.0x its cost elsewhere (0.28 vs 0.14 nats), e.g. 'He likes to run and' -> ' jump'.
           Head 2 still matters more after ' and' (0.51 vs 0.26) and ',' (0.51 vs 0.29); head 3 leads
           after ' to' (0.41 vs 0.22) and ' of' (0.36 vs 0.28).
  Removed  Replacing its votes by their average raises the KL by 0.16 nats; top-1 agreement with the
           model falls from 69% to 61%. Heads 2 and 3 overlap: with head 2 removed as well, head 3's
           cost rises to 0.24, so single-head costs do not add up.
  Model    Ablating the earlier-token part of the model's head 3 costs 0.32 nats; the program head
           carries 49% of that. The model head is even more focused on linking words (3.1x): after
           ' and' it matters 4.5x more than this head (1.18 vs 0.26 nats). It also looks further back
           (mean distance 9.6) and looks at the current token less often (19%).
"""
from .attention import Head
from .tables import head3 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
