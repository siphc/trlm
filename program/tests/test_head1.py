"""CLAIM (head1.py): RECALL's votes from earlier tokens point the same way as model head 1's effect from earlier
tokens (median weighted correlation >= 0.5; TRAIN 0.64), better than the same votes with uniform attention over
earlier tokens (TRAIN 0.55). Its effect on the output is weaker than the head's (TRAIN 0.079 vs 0.41 nats)."""
from program.tests import measure as M


def test_agrees_with_model_head1(P, W, lp, ablated):
    rule = M.compare_with_head(P, W, lp, ablated["h1"], lambda t: M.earlier_votes(P, "head1", t))
    flat = M.compare_with_head(P, W, lp, ablated["h1"],
                               lambda t: M.earlier_votes(P, "head1", t, attention=M.uniform_attention(len(t))))
    print(rule, flat)
    assert rule["agreement"] >= 0.5
    assert flat["agreement"] < rule["agreement"]
    assert rule["shift_program"] < rule["shift_model"]
