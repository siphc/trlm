"""CLAIMS (head1.py): RECALL's votes from earlier tokens point the same way as model head 1's effect from earlier
tokens (median weighted correlation >= 0.5; TRAIN 0.62), better than the same votes with uniform attention over
earlier tokens (TRAIN 0.53). Its effect on the output is weaker than the head's (TRAIN 0.11 vs 0.41 nats)."""
from program.tests import measure as M


def test_agrees_with_model_head1(P, W, lp, ablated):
    rule = M.compare_with_head(P, W, lp, ablated["h1"], lambda t: M.earlier_votes(P, "head1", t))
    flat = M.compare_with_head(P, W, lp, ablated["h1"],
                               lambda t: M.earlier_votes(P, "head1", t, attention=M.uniform_attention(len(t))))
    print(rule, flat)
    assert rule["agreement"] >= 0.5
    assert flat["agreement"] < rule["agreement"]
    assert rule["shift_program"] < rule["shift_model"]


def test_vote_has_fixed_size(P, W, lp, ablated, tokens):
    """CLAIM (head1.py, SIZE): where RECALL's raw (not rescaled) vote from earlier tokens is largest (top tenth of
    positions), model head 1's earlier-token effect grows less than that raw vote, relative to the middle fifth
    (TRAIN: model 1.93x, raw vote 2.86x), and the fixed-size vote is closer to the model's growth (TRAIN 2.38x)."""
    import numpy as np

    from program import head1
    from program.readers import earlier_only

    s = P.STATS

    def size(x, w):
        mu = (x * w).sum(-1, keepdims=True)
        return np.sqrt((w * (x - mu) ** 2).sum(-1))

    m, raw, fixed = [], [], []
    for t, p, pa in zip(tokens, lp, ablated["h1"]):
        a = head1.attention(t, s)
        w = np.exp(p)
        m.append(size(p - pa, w))
        raw.append(size(earlier_only(a) @ head1.recalled_votes(t, s), w))
        fixed.append(size(head1.earlier_votes(a, t, s), w))
    m, raw, fixed = map(np.concatenate, (m, raw, fixed))
    q = np.quantile(raw, [0.4, 0.6, 0.9])
    top, mid = raw >= q[2], (raw >= q[0]) & (raw < q[1])
    growth = {k: float(x[top].mean() / x[mid].mean()) for k, x in (("model", m), ("raw", raw), ("fixed", fixed))}
    print(growth)
    assert growth["model"] < growth["raw"]
    assert abs(growth["fixed"] - growth["model"]) < abs(growth["raw"] - growth["model"])
