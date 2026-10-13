"""Claim tests for program/head3.py (carries a phrase across linking words).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v10 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound.
"""
from measurements import at_least, at_most, claim_numbers, within


def test_abstains_on_a_third():
    """Head 3 puts 37% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head3.abstain"], 0.3685)


def test_looks_a_little_further_back():
    """16% of its attention is on the previous token, 34% on 2 to 8 back, 12% further; mean distance
    5.5 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head3.previous"], 0.1628)
    within(m["head3.near"], 0.3438)
    within(m["head3.far"], 0.1249)
    within(m["head3.mean_distance"], 5.4848)


def test_never_more_than_40_back():
    """Under 0.1% of its attention goes more than 40 tokens back."""
    at_most(claim_numbers()["head3.beyond_40"], 0.001)


def test_matters_most_after_linking_words():
    """Right after a linking word its removal cost is 2.0x its cost elsewhere (at least -25%)."""
    at_least(claim_numbers()["head3.after_linker_ratio"], 2.0242)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.16 nats, and top-1 agreement falls by
    8.1 points, from 69% to 61% (both +-30%)."""
    m = claim_numbers()
    within(m["head3.cost"], 0.1564)
    within(m["top1"] - m["head3.top1_without"], 0.6897 - 0.6085)


def test_matches_half_of_model_head3():
    """Ablating the earlier-token part of the model's head 3 costs 0.32 nats; the program head carries
    49% of it (both +-30%); the model head is even more focused after linking words (3.1x, at least
    -25%)."""
    m = claim_numbers()
    within(m["head3.model_cost"], 0.3224)
    within(m["head3.share_of_model"], 0.4851)
    at_least(m["head3.model_after_linker_ratio"], 3.0645)
