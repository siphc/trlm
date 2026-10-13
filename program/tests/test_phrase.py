"""CLAIM (phrase.py): PHRASE stands in for part of heads 0 and 2. Its votes agree with the model's change when the
earlier-token parts of heads 0 and 2 are replaced by their mean (median weighted correlation >= 0.3; TRAIN 0.40),
and at least twice as much as with head 1's (TRAIN 0.08), the long-range head."""
from program import phrase
from program.tests import measure as M


def test_phrase_is_heads_0_and_2(P, W, lp, ablated):
    agree = {k: M.vote_agreement(W, lp, ablated[k], lambda t: phrase.votes(t, P.STATS)) for k in ["h02", "h1"]}
    print(agree)
    assert agree["h02"] >= 0.3
    assert agree["h02"] >= 2 * agree["h1"]
