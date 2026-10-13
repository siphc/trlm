"""Claim tests for program/head0.py (passes on the last few tokens and finishes split words).

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound. The model head's
attention share quoted in head0.py comes from analysis/claims_measure.py and is not tested here.
"""
from measurements import at_least, at_most, claim_numbers, phrase, within

SHIRT = ["He", " had", " a", " new", " sh", "ir"]   # 'He had a new shir', as the tokenizer splits it


def test_abstains_on_more_than_half_its_attention():
    """Head 0 puts 58% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head0.abstain"], 0.5770)


def test_looks_close_by():
    """19% of its attention is on the previous token, 17% on 2 to 8 back, 5% further; mean distance
    4.2 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head0.previous"], 0.1933)
    within(m["head0.near"], 0.1748)
    within(m["head0.far"], 0.0549)
    within(m["head0.mean_distance"], 4.2097)


def test_never_more_than_48_back():
    """Under 0.1% of its attention goes more than 48 tokens back."""
    at_most(claim_numbers()["head0.beyond_48"], 0.001)


def test_finishes_split_words():
    """Inside words split into pieces its removal cost is 2.6x its cost elsewhere (at least -25%), 0.34 vs
    0.13 nats (both +-30%), more than any other head's cost there (next: head 3, 0.30, +-30%)."""
    m = claim_numbers()
    at_least(m["head0.inside_word_ratio"], 2.6473)
    within(m["head0.cost_inside_word"], 0.3350)
    within(m["head0.cost_not_inside_word"], 0.1265)
    ranked = sorted(range(4), key=lambda h: -m[f"head{h}.cost_inside_word"])
    assert ranked[:2] == [0, 3], ranked
    within(m["head3.cost_inside_word"], 0.3014)


def test_shirt_example():
    """After 'He had a new shir' the program gives 't' 0.91 (at least -25%), and 0.11 with this head's votes
    replaced by their average (at most +25%)."""
    r = phrase(SHIRT, "t", 0)
    at_least(r["p"], 0.9089)
    at_most(r["p_without"], 0.1120 * 1.25)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.15 nats (+-30%), the least of the four heads,
    and top-1 agreement falls by 6.6 points, from 69% to 63% (+-30%)."""
    m = claim_numbers()
    within(m["head0.cost"], 0.1534)
    assert m["head0.cost"] == min(m[f"head{h}.cost"] for h in range(4))
    within(m["top1"] - m["head0.top1_without"], 0.6936 - 0.6279)


def test_matches_a_third_of_model_head0():
    """Ablating the earlier-token part of the model's head 0 costs 0.46 nats (+-30%); the program head
    carries a third of it (33%, +-30%); the model head is also focused inside words (2.0x, at least -25%).
    The model gives 't' after 'He had a new shir' 0.98 (at least -25%), and under 0.01 without its head 0."""
    m = claim_numbers()
    within(m["head0.model_cost"], 0.4649)
    within(m["head0.share_of_model"], 0.3300)
    at_least(m["head0.model_inside_word_ratio"], 2.0380)
    r = phrase(SHIRT, "t", 0)
    at_least(r["model_p"], 0.9751)
    at_most(r["model_p_without"], 0.01)
