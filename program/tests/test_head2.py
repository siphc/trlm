"""CLAIM (head2.py): LOCAL-2's votes from earlier tokens point the same way as model head 2's effect from earlier
tokens (median weighted correlation >= 0.25; TRAIN 0.35), and its attention rule is what makes them agree: with
uniform attention the agreement falls below half (TRAIN 0.09). Removing it moves the program's output less than a
quarter as far as ablating the head moves the model's (TRAIN 0.11 vs 0.49 nats)."""
from program.tests import measure as M


def test_agrees_with_model_head2(P, W, lp, ablated):
    rule = M.compare_with_head(P, W, lp, ablated["h2"], lambda t: M.earlier_votes(P, "head2", t))
    flat = M.compare_with_head(P, W, lp, ablated["h2"],
                               lambda t: M.earlier_votes(P, "head2", t, attention=M.uniform_attention(len(t))))
    print(rule, flat)
    assert rule["agreement"] >= 0.25
    assert flat["agreement"] < 0.5 * rule["agreement"]
    assert rule["shift_program"] < 0.25 * rule["shift_model"]
