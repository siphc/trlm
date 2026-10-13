"""Claim tests for program/follow.py (FOLLOW: the guess from the current token alone).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound.
"""
import numpy as np

from measurements import at_least, at_most, claim_numbers, follow_exceptions, within


def test_follow_alone():
    """On its own FOLLOW is 1.39 nats from the model and picks the model's top token 34% of the time
    (both +-30%)."""
    m = claim_numbers()
    within(m["kl_follow"], 1.3932)
    within(m["top1_follow"], 0.3351)


def test_fixed_phrases():
    """After ' upon' FOLLOW alone gives ' a' 0.92 (at least -25%), and ' a' is the model's top token at all
    held-out ' upon's (at least 95% of them; 140 occurrences, at least 50 needed), as is FOLLOW's top token.
    After ' wanted' FOLLOW alone gives ' to' 0.87 (at least -25%) and the model 0.87 on average (+-30%;
    119 occurrences, at least 50 needed)."""
    m = claim_numbers()
    assert m["upon_n"] >= 50, m["upon_n"]
    at_least(m["upon_follow_p_a"], 0.9188)
    assert m["upon_model_top_is_a"] >= 0.95, m["upon_model_top_is_a"]
    assert m["upon_follow_matches_model"] >= 0.95, m["upon_follow_matches_model"]
    assert m["wanted_n"] >= 50, m["wanted_n"]
    at_least(m["wanted_follow_p_to"], 0.8673)
    within(m["wanted_model_p_to"], 0.8690)


def test_once_needs_the_heads():
    """After 'Once' FOLLOW alone gives ' upon' 0.38 against the model's 0.88 (both +-30%; 164 occurrences,
    at least 50 needed). The heads supply the rest, mostly head 2: with its votes replaced by their average
    ' upon' gets 0.51 (at most +25%), the lowest of the four heads. KL there is 0.66 for FOLLOW alone (+-30%)
    and 0.03 for the whole program (at most +25%)."""
    m = claim_numbers()
    assert m["once_n"] >= 50, m["once_n"]
    within(m["once_follow_p_upon"], 0.3790)
    within(m["once_model_p_upon"], 0.8826)
    at_most(m["head2.once_p_upon_without"], 0.5102 * 1.25)
    assert m["head2.once_p_upon_without"] == min(m[f"head{h}.once_p_upon_without"] for h in range(4))
    within(m["once_kl_follow"], 0.6626)
    at_most(m["once_kl"], 0.0281 * 1.25)


def test_exceptions_are_the_largest_part():
    """FOLLOW has 31k token-pair exceptions (31,158, +-10%). Replacing them by their average raises the KL by
    0.89 nats (+-30%), more than replacing its class table (CLASS_FOLLOW) does (0.75, +-30%) and over four
    times the cost of any one head (at most 0.21)."""
    m = claim_numbers()
    within(np.count_nonzero(follow_exceptions()), 31158, rel=0.10)
    within(m["exceptions_cost"], 0.8860)
    within(m["class_follow_cost"], 0.7504)
    assert m["exceptions_cost"] > m["class_follow_cost"], (m["exceptions_cost"], m["class_follow_cost"])
    heads = max(m[f"head{h}.cost"] for h in range(4))
    assert m["exceptions_cost"] > 4 * heads, (m["exceptions_cost"], heads)
