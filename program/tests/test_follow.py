"""CLAIM (follow.py): FOLLOW is what the model predicts from the current token alone. On its own it is much closer
to the model with the earlier-token parts of all four heads replaced by their mean than to the full model:
KL at most 1.35 nats and at most 0.8 times its KL to the full model (TRAIN: 1.13 vs 1.67)."""
import numpy as np

from program.tests import measure as M


def test_follow_is_the_context_blind_model(P, lp, ablated, tokens):
    others = tuple(m.NAME for m in P.MECHANISMS if m.NAME != "follow")
    q = [P.predict(t, off=others) for t in tokens]
    to_blind = np.mean([M.kl(p, qq).mean() for p, qq in zip(ablated["all"], q)])
    to_full = np.mean([M.kl(p, qq).mean() for p, qq in zip(lp, q)])
    print(to_blind, to_full)
    assert to_blind <= 1.35
    assert to_blind <= 0.8 * to_full
