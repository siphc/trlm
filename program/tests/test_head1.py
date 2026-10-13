"""Claim tests for program/head1.py (recalls words from far back and votes to repeat them).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound. The model head's
attention numbers quoted in head1.py come from analysis/claims_measure.py and are not tested here.
"""
from measurements import at_least, at_most, claim_numbers, within


def test_abstains_on_under_a_third():
    """Head 1 puts 28% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head1.abstain"], 0.2849)


def test_looks_far_back():
    """Mean distance 34 tokens and 61% of its attention more than 8 back (both +-30%); 0.8% on the
    previous token (at most +25%); under 0.1% more than 128 back."""
    m = claim_numbers()
    within(m["head1.mean_distance"], 33.6896)
    within(m["head1.far"], 0.6098)
    at_most(m["head1.previous"], 0.0082 * 1.25)
    at_most(m["head1.beyond_128"], 0.001)


def test_prefers_names_and_votes_for_what_it_sees():
    """Names get 3.3x their share of its attention; a token it attends to votes for itself typically
    18 nats more than for the other words of its TARGET class (both at least -25%)."""
    m = claim_numbers()
    at_least(m["head1.name_enrichment"], 3.2629)
    at_least(m["head1.self_vote"], 17.70)


def test_recall_positions():
    """Recall positions are one position in five (21%, +-30%). There head 1 raises log q(model's top token)
    by 1.03 nats, against 0.15 elsewhere (both +-30%): 6.9x (at least -25%)."""
    m = claim_numbers()
    within(m["recall_share"], 0.2100)
    within(m["head1.gain_recall"], 1.0286)
    within(m["head1.gain_other"], 0.1491)
    at_least(m["head1.recall_ratio"], 6.8990)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.18 nats (+-30%); at recall positions
    top-1 agreement falls by 23 points, from 60% to 38% (+-30%)."""
    m = claim_numbers()
    within(m["head1.cost"], 0.1827)
    within(m["recall_top1"] - m["head1.recall_top1_without"], 0.6040 - 0.3764)


def test_matches_44_percent_of_model_head1():
    """Ablating the earlier-token part of the model's head 1 costs 0.42 nats; the program head carries
    44% of it (both +-30%); the model head is also focused on recall positions: 1.49 vs 0.29 nats (both
    +-30%), 5.1x (at least -25%)."""
    m = claim_numbers()
    within(m["head1.model_cost"], 0.4165)
    within(m["head1.share_of_model"], 0.4388)
    within(m["head1.model_gain_recall"], 1.4884)
    within(m["head1.model_gain_other"], 0.2936)
    at_least(m["head1.model_recall_ratio"], 5.0702)
