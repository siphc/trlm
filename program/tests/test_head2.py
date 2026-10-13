"""Claim tests for program/head2.py (reads the last few words; opens quotes after 'said,').

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v10 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound.
"""
from measurements import at_least, at_most, claim_numbers, within


def test_abstains_on_a_third():
    """Head 2 puts 36% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head2.abstain"], 0.3646)


def test_reads_the_last_few_tokens():
    """25% of its attention is on the previous token, 33% on 2 to 8 back, 6% further; mean distance
    3.6 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head2.previous"], 0.2480)
    within(m["head2.near"], 0.3289)
    within(m["head2.far"], 0.0585)
    within(m["head2.mean_distance"], 3.6477)


def test_essentially_never_more_than_32_back():
    """Under 0.1% of its attention goes more than 32 tokens back."""
    at_most(claim_numbers()["head2.beyond_32"], 0.001)


def test_opens_quotes_after_said():
    """After a speech verb and a comma the program gives the quote ' "' 0.86 with head 2 (at least -25%)
    and 0.13 without (at most +25%); at least 50 such positions. Right after a speech verb its removal
    cost is 3.9x its cost elsewhere (at least -25%)."""
    m = claim_numbers()
    assert m["said_comma_n"] >= 50, m["said_comma_n"]
    at_least(m["said_comma_p_quote"], 0.8625)
    at_most(m["head2.said_comma_p_quote_without"], 0.1314 * 1.25)
    at_least(m["head2.after_speech_verb_ratio"], 3.8700)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.27 nats, the most of the four heads, and
    top-1 agreement falls by 12.5 points, from 69% to 56% (both +-30%)."""
    m = claim_numbers()
    within(m["head2.cost"], 0.2683)
    assert m["head2.cost"] == max(m[f"head{h}.cost"] for h in range(4))
    within(m["top1"] - m["head2.top1_without"], 0.6897 - 0.5649)


def test_matches_55_percent_of_model_head2():
    """Ablating the earlier-token part of the model's head 2 costs 0.49 nats; the program head carries
    55% of it (both +-30%), the closest match of the four; the model head is also focused right after
    speech verbs (3.2x, at least -25%)."""
    m = claim_numbers()
    within(m["head2.model_cost"], 0.4905)
    within(m["head2.share_of_model"], 0.5470)
    assert m["head2.share_of_model"] == max(m[f"head{h}.share_of_model"] for h in range(4))
    at_least(m["head2.model_after_speech_verb_ratio"], 3.1829)
