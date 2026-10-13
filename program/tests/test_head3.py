"""Claim tests for program/head3.py (carries a phrase across linking words such as ' and', ' of' and ' to').

Numbers: analysis/claims_measure.py on the 200 held-out claim windows (v11 tables); definitions:
measurements.py. Tolerances: +-30% around a stated value, 25% slack on a one-sided bound. The model head's
attention numbers quoted in head3.py come from analysis/claims_measure.py and are not tested here.
"""
from measurements import at_least, at_most, claim_numbers, phrase, within

JUMP = ["He", " likes", " to", " run", " and"]   # 'He likes to run and', as the tokenizer splits it


def test_abstains_on_under_a_third():
    """Head 3 puts 30% of its attention on the current token, where it casts no vote (+-30%)."""
    within(claim_numbers()["head3.abstain"], 0.2987)


def test_reads_the_last_few_tokens():
    """25% of its attention is on the previous token, 35% on 2 to 8 back, 10% further; mean distance
    4.3 tokens (all +-30%)."""
    m = claim_numbers()
    within(m["head3.previous"], 0.2533)
    within(m["head3.near"], 0.3507)
    within(m["head3.far"], 0.0974)
    within(m["head3.mean_distance"], 4.3143)


def test_never_more_than_40_back():
    """Under 0.1% of its attention goes more than 40 tokens back."""
    at_most(claim_numbers()["head3.beyond_40"], 0.001)


def test_matters_most_after_linking_words():
    """Right after a linking word its removal cost is 2.0x its cost elsewhere (at least -25%), 0.36 vs
    0.18 nats (both +-30%). After 'He likes to run and' the program gives ' jump' 0.15 (at least -25%),
    and under 0.01 with this head's votes replaced by their average."""
    m = claim_numbers()
    at_least(m["head3.after_linker_ratio"], 1.9779)
    within(m["head3.cost_after_linker"], 0.3628)
    within(m["head3.cost_not_after_linker"], 0.1834)
    r = phrase(JUMP, " jump", 3)
    at_least(r["p"], 0.1524)
    at_most(r["p_without"], 0.01)


def test_most_important_head_after_and_of_to():
    """It is the most important head after ' and' (0.52 nats; next head 2, 0.28), ' of' (0.49; head 0, 0.36)
    and ' to' (0.34; head 0, 0.23); after ',' heads 2, 3 and 1 matter about equally: 0.36, 0.34 and 0.31
    (all values +-30%; the largest of the three at most 1.3x the smallest)."""
    m = claim_numbers()
    stated = {"and": {3: 0.5245, 2: 0.2810}, "of": {3: 0.4875, 0: 0.3595}, "to": {3: 0.3365, 0: 0.2271}}
    for token, heads in stated.items():
        ranked = sorted(range(4), key=lambda h: -m[f"head{h}.cost_after_{token}"])
        assert ranked[:2] == list(heads), (token, ranked)
        for h, value in heads.items():
            within(m[f"head{h}.cost_after_{token}"], value)
    comma = [m[f"head{h}.cost_after_comma"] for h in (2, 3, 1)]
    for value, stated in zip(comma, (0.3590, 0.3358, 0.3063)):
        within(value, stated)
    assert max(comma) <= 1.3 * min(comma), comma


def test_removal_cost():
    """Replacing its votes by their average raises the KL by 0.21 nats (+-30%), the most of the four heads,
    and top-1 agreement falls by 10.9 points, from 69% to 58% (+-30%). With head 2 removed as well, its cost
    rises to 0.30 (+-30%)."""
    m = claim_numbers()
    within(m["head3.cost"], 0.2055)
    assert m["head3.cost"] == max(m[f"head{h}.cost"] for h in range(4))
    within(m["top1"] - m["head3.top1_without"], 0.6936 - 0.5846)
    within(m["head3.cost_with_head2_removed"], 0.3034)
    assert m["head3.cost_with_head2_removed"] > m["head3.cost"]


def test_matches_64_percent_of_model_head3():
    """Ablating the earlier-token part of the model's head 3 costs 0.32 nats; the program head carries
    64% of it (both +-30%), the most of the four. The model head is also the model's most important head
    after ' and', ' of' and ' to'. It is even more focused after linking words (3.1x, at least -25%); after
    ' and' it costs 1.18 nats (+-30%), 2.3x this head's cost there (at least -25%)."""
    m = claim_numbers()
    within(m["head3.model_cost"], 0.3224)
    within(m["head3.share_of_model"], 0.6376)
    assert m["head3.share_of_model"] == max(m[f"head{h}.share_of_model"] for h in range(4))
    for token in ("and", "of", "to"):
        model = max(range(4), key=lambda h: m[f"head{h}.model_cost_after_{token}"])
        assert model == 3, (token, model)
    at_least(m["head3.model_after_linker_ratio"], 3.0645)
    within(m["head3.model_cost_after_and"], 1.1815)
    at_least(m["head3.model_cost_after_and"] / m["head3.cost_after_and"], 2.2526)
