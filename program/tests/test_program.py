"""Claim tests for program/__init__.py (the whole algorithm).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value (+-15% for the heads' total gain), 25% slack on a one-sided bound.
"""
from measurements import at_least, at_most, claim_numbers, within


def test_scores_are_follow_plus_head_votes():
    """predict() is exactly log_softmax(FOLLOW + the four heads' votes) (max abs difference < 1e-9)."""
    at_most(claim_numbers()["decomposition_error"], 1e-9)


def test_fidelity():
    """KL(model || program) is 0.45 nats (one-sided: at most +25%); top-1 agreement is 69% (at least -25%)."""
    m = claim_numbers()
    at_most(m["kl"], 0.4542 * 1.25)
    at_least(m["top1"], 0.6936)


def test_heads_remove_two_thirds_of_follow_alone_kl():
    """FOLLOW alone is 1.39 nats from the model; adding the four heads removes 0.94 nats (+-15%: tight enough
    that losing any one head fails this test; with head 0, the least important, mean-ablated the gain is 0.79),
    two thirds of it (0.67, +-10%)."""
    m = claim_numbers()
    within(m["heads_gain"], 0.9389, rel=0.15)
    within(m["heads_gain"] / m["kl_follow"], 0.6740, rel=0.10)
