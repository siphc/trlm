"""HEAD 0 of the model's single attention layer: passes on the last few tokens and finishes split words.

CLAIM (measured on 200 held-out windows by analysis/claims_measure.py; tests/test_head0.py):
  Role     Head 0 abstains on 58% of its attention (the current token: no vote, see attention.py).
           The rest comes from close by: 19% on the previous token, 17% on 2 to 8 tokens back, 5%
           further (mean distance 4.2 tokens), and nothing (under 0.1%) more than 48 tokens back.
  Words    It matters most inside words that the tokenizer splits into pieces: its removal cost there
           is 2.6x its cost elsewhere (0.34 vs 0.13 nats), more than any other head's (next: head 3,
           0.30). After 'He had a new shir' (' sh' 'ir', as in 'shirt') the program gives 't' 0.91,
           and 0.11 without this head.
  Removed  Replacing its votes by their average raises the KL by 0.15 nats, the least of the four
           heads; top-1 agreement with the model falls from 69% to 63%.
  Model    Ablating the earlier-token part of the model's head 0 costs 0.46 nats; the program head
           carries a third of that, with the same inside-word focus (model: 2.0x; without its head 0
           the model's 't' after 'He had a new shir' falls from 0.98 to under 0.01). The model head
           also spends half its attention (49%) on the current token, where it carries that token's
           own vote; in the program FOLLOW does that part of the job.
"""
from .attention import Head
from .tables import head0 as tables

HEAD = Head(tables)


def votes(tokens):
    """Scores this head adds for every next token at each position, shape (n, VOCAB_SIZE)."""
    return HEAD.votes(tokens)
