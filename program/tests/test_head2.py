"""Claim tests for program/head2.py (completes phrases from the last few words; opens quotes after 'said,').

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound. A head is
"needed" for a phrase when replacing its votes by their average (program) or mean-ablating its earlier-token
part (model) leaves the token at most 0.1, and "not needed" when the token keeps at least 0.7.
"""
from measurements import at_least, at_most, claim_numbers, phrase, within

MORAL = ["The", " moral", " of", " the"]        # 'The moral of the', as the tokenizer splits it
PROBLEM = ["She", " had", " a", " pro", "ble"]  # 'She had a proble'
EYES = ["He", " closed", " his"]                # 'He closed his'


def test_abstains_on_a_third():
    """Head 2 puts 36% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head2.abstain"], 0.3562)


def test_reads_the_last_few_tokens():
    """17% of its attention is on the previous token, 37% on 2 to 8 back, 10% further; mean distance
    5.0 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head2.previous"], 0.1741)
    within(m["head2.near"], 0.3666)
    within(m["head2.far"], 0.1031)
    within(m["head2.mean_distance"], 4.9775)


def test_never_more_than_40_back():
    """Under 0.1% of its attention goes more than 40 tokens back."""
    at_most(claim_numbers()["head2.beyond_40"], 0.001)


def test_completes_phrases():
    """After 'The moral of the' the program gives ' story' 0.97 (at least -25%), and 0.03 with this head's
    votes replaced by their average (at most +25%)."""
    r = phrase(MORAL, " story", 2)
    at_least(r["p"], 0.9691)
    at_most(r["p_without"], 0.0267 * 1.25)


def test_opens_quotes_after_said():
    """After a speech verb and a comma the opening quote ' "' is the model's top token at 94% of 247
    positions (at least -25%; at least 50 positions needed); the program gives it 0.87 with head 2 (at least
    -25%) and 0.26 without (at most +25%). Right after a speech verb its removal cost is 3.5x its cost
    elsewhere (at least -25%)."""
    m = claim_numbers()
    assert m["said_comma_n"] >= 50, m["said_comma_n"]
    at_least(m["said_comma_model_quote"], 0.9433)
    at_least(m["said_comma_p_quote"], 0.8686)
    at_most(m["head2.said_comma_p_quote_without"], 0.2570 * 1.25)
    at_least(m["head2.after_speech_verb_ratio"], 3.5461)


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.19 nats, and top-1 agreement falls by
    9.5 points, from 69% to 60% (both +-30%)."""
    m = claim_numbers()
    within(m["head2.cost"], 0.1907)
    within(m["top1"] - m["head2.top1_without"], 0.6936 - 0.5987)


def test_matches_39_percent_of_model_head2():
    """Ablating the earlier-token part of the model's head 2 costs 0.49 nats; the program head carries
    39% of it (both +-30%); the model head is also focused right after speech verbs (3.2x, at least -25%).
    The model gives ' story' after 'The moral of the' 0.91 (at least -25%), and under 0.01 without its head 2."""
    m = claim_numbers()
    within(m["head2.model_cost"], 0.4905)
    within(m["head2.share_of_model"], 0.3888)
    at_least(m["head2.model_after_speech_verb_ratio"], 3.1829)
    r = phrase(MORAL, " story", 2)
    at_least(r["model_p"], 0.9143)
    at_most(r["model_p_without"], 0.01)


def test_some_model_head2_work_sits_in_other_heads():
    """Inside split words the model head's cost is 0.94 nats (+-30%), the most of the model's heads; this
    head's is 0.27 (+-30%). The model needs its head 2 for 'm' after 'She had a proble' and ' eyes' after
    'He closed his'; the program does not need head 2 for them but needs head 0 ('m') and head 3 (' eyes')."""
    m = claim_numbers()
    within(m["head2.model_cost_inside_word"], 0.9396)
    assert m["head2.model_cost_inside_word"] == max(m[f"head{h}.model_cost_inside_word"] for h in range(4))
    within(m["head2.cost_inside_word"], 0.2672)
    for tokens, nxt, other in ((PROBLEM, "m", 0), (EYES, " eyes", 3)):
        r = phrase(tokens, nxt, 2)
        assert r["model_p_without"] <= 0.1 and r["p_without"] >= 0.7, (tokens, r)
        assert phrase(tokens, nxt, other)["p_without"] <= 0.1, (tokens, other)
