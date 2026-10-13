"""Claim tests for program/head0.py (passes on the last few tokens and finishes split words).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v10 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound.
"""
from measurements import at_least, at_most, claim_numbers, within


def test_abstains_on_half_its_attention():
    """Head 0 puts 52% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head0.abstain"], 0.5234)


def test_looks_close_by():
    """18% of its attention is on the previous token, 20% on 2 to 8 back, 10% further; mean distance
    5.4 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head0.previous"], 0.1830)
    within(m["head0.near"], 0.1973)
    within(m["head0.far"], 0.0964)
    within(m["head0.mean_distance"], 5.3504)


def test_never_more_than_48_back():
    """Under 0.1% of its attention goes more than 48 tokens back."""
    at_most(claim_numbers()["head0.beyond_48"], 0.001)


def test_finishes_split_words():
    """Inside words split into pieces its removal cost is 3.0x its cost elsewhere (at least -25%)."""
    at_least(claim_numbers()["head0.inside_word_ratio"], 2.9698)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.15 nats, and top-1 agreement falls by
    6.3 points, from 69% to 63% (both +-30%)."""
    m = claim_numbers()
    within(m["head0.cost"], 0.1469)
    within(m["top1"] - m["head0.top1_without"], 0.6897 - 0.6265)


def test_matches_a_third_of_model_head0():
    """Ablating the earlier-token part of the model's head 0 costs 0.46 nats (+-30%); the program head
    carries a third of it (32%, +-30%); the model head is also focused inside words (2.0x, at least -25%)."""
    m = claim_numbers()
    within(m["head0.model_cost"], 0.4649)
    within(m["head0.share_of_model"], 0.3160)
    at_least(m["head0.model_inside_word_ratio"], 2.0380)
