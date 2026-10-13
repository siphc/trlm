"""HEAD 3 of the model's single attention layer: carries a phrase across linking words such as ' and', ' of' and ' to'.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head3.py):
  Role     Head 3 reads the last few tokens: 25% of its attention on the previous token, 35% on 2 to
           8 back, 10% further (mean distance 4.3), nothing (under 0.1%) more than 40 back. It
           abstains on 30%.
  Linkers  It matters most right after linking words (' and', ',', ' to', ' of', ...): its removal
           cost there is 2.0x its cost elsewhere (0.36 vs 0.18 nats). After 'He likes to run and' the
           program gives ' jump' 0.15, and under 0.01 without this head. It is the most important head
           after ' and' (0.52 nats; next head 2, 0.28), ' of' (0.49; head 0, 0.36) and ' to' (0.34;
           head 0, 0.23); after ',' heads 2, 3 and 1 matter about equally (0.36, 0.34, 0.31).
  Removed  The most important head, narrowly: replacing its votes by their average raises the KL by
           0.21 nats (head 2: 0.19); top-1 agreement with the model falls from 69% to 58%. Heads 2
           and 3 overlap: with head 2 removed as well, head 3's cost rises to 0.30, so single-head
           costs do not add up.
  Model    Ablating the earlier-token part of the model's head 3 costs 0.32 nats; the program head
           carries 64% of that, the closest match of the four. The model head is also the model's most
           important head after ' and', ' of' and ' to', and it is even more focused on linking words
           (3.1x): after ' and' it matters 2.3x more than this head (1.18 vs 0.52 nats). It also looks
           further back (mean distance 9.6) and looks at the current token less often (19%).
"""
from .attention import Head
from .tables import head3 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
