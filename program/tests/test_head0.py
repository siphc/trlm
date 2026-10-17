"""CLAIM (head0.py): LOCAL-0's votes from earlier tokens point the same way as model head 0's effect from earlier
tokens (median weighted correlation >= 0.18; TRAIN 0.28), and its attention rule is what makes them agree: with
uniform attention over earlier tokens the agreement falls below half of that (TRAIN 0.02). It captures only a small
part of the head: removing it moves the program's output less than a quarter as far as ablating the head moves the
model's (TRAIN 0.030 vs 0.46 nats)."""
from program.tests import measure as M


def test_agrees_with_model_head0(P, W, lp, ablated):
    rule = M.compare_with_head(P, W, lp, ablated["h0"], lambda t: M.earlier_votes(P, "head0", t))
    flat = M.compare_with_head(P, W, lp, ablated["h0"],
                               lambda t: M.earlier_votes(P, "head0", t, attention=M.uniform_attention(len(t))))
    print(rule, flat)
    assert rule["agreement"] >= 0.18
    assert flat["agreement"] < 0.5 * rule["agreement"]
    assert rule["shift_program"] < 0.25 * rule["shift_model"]
