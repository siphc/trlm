"""CLAIM (program/__init__.py): the five mechanisms together are within 0.62 nats of the model on held-out text,
and each one is needed: switching it off raises KL by at least 0.05 nats (FOLLOW by at least 1.0).
Measured on 64 TRAIN windows (v1.3): 0.547; rises FOLLOW 1.36, PHRASE 0.14, LOCAL-2 0.13, RECALL 0.12, LOCAL-0 0.066."""
from program.tests import measure as M


def test_close_to_model_and_every_mechanism_needed(P, W, lp):
    rise = M.removal_rise(P, W, lp, [m.NAME for m in P.MECHANISMS])
    print(rise)
    assert rise["all"] <= 0.62
    assert rise["follow"] >= 1.0
    for name in ["head0", "head1", "head2", "phrase"]:
        assert rise[name] >= 0.05, name
