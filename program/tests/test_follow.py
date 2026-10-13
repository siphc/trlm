"""Claim tests for program/follow.py (FOLLOW: the guess from the current token alone).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v10 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound.
"""
from measurements import at_least, claim_numbers, within


def test_follow_alone():
    """On its own FOLLOW is 1.40 nats from the model and picks the model's top token 34% of the time
    (both +-30%)."""
    m = claim_numbers()
    within(m["kl_follow"], 1.3988)
    within(m["top1_follow"], 0.3385)


def test_fixed_phrase_after_upon():
    """After ' upon' FOLLOW alone gives ' a' 0.92 (at least -25%), and its top token is the model's top
    token at all held-out ' upon's (at least 95% of them; 140 occurrences, at least 50 needed)."""
    m = claim_numbers()
    assert m["upon_n"] >= 50, m["upon_n"]
    at_least(m["upon_follow_p_a"], 0.9239)
    assert m["upon_follow_matches_model"] >= 0.95, m["upon_follow_matches_model"]


def test_exceptions_are_the_largest_part():
    """Replacing FOLLOW's token-pair exceptions by their average raises the KL by 1.43 nats (+-30%),
    more than removing all four heads does (0.93)."""
    m = claim_numbers()
    within(m["exceptions_cost"], 1.4252)
    assert m["exceptions_cost"] > m["heads_gain"], (m["exceptions_cost"], m["heads_gain"])
