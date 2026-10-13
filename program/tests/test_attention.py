"""Claim tests for program/attention.py (the abstention rule: heads cast no vote with the current token).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v10 tables); definitions:
measurements.py. The per-head abstention shares are tested in test_head0.py ... test_head3.py.
"""
from measurements import at_least, at_most, claim_numbers


def test_first_position_is_follow_alone():
    """At the first token of a text every head abstains fully, so the prediction there is exactly
    FOLLOW alone (max abs difference < 1e-9)."""
    at_most(claim_numbers()["first_position_error"], 1e-9)


def test_voting_with_the_current_token_would_hurt():
    """Letting the heads also vote with the current token raises KL to the model from 0.47 to 1.89
    nats: a rise of 1.43 nats (one-sided: at least -25%)."""
    at_least(claim_numbers()["self_votes_cost"], 1.4260)
