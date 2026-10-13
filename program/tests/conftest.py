"""Shared fixtures for the claim tests: the program (with its corpus tables), held-out claim windows, and the
model's log-probs with and without the earlier-token part of each head (harness.claims)."""
import numpy as np
import pytest

from harness import claims

N_WINDOWS = 32
ABLATIONS = {"h0": [0], "h1": [1], "h2": [2], "h3": [3], "h02": [0, 2], "all": [0, 1, 2, 3]}


@pytest.fixture(scope="session")
def P():
    return claims.program()


@pytest.fixture(scope="session")
def W():
    return claims.claim_windows()[:N_WINDOWS]


@pytest.fixture(scope="session")
def lp(W):
    return claims.model_logprobs(W)


@pytest.fixture(scope="session")
def ablated(W):
    """Model log-probs with the listed heads' earlier-token parts replaced by their mean."""
    return {k: claims.model_logprobs(W, ablate_heads=h, ablation="earlier") for k, h in ABLATIONS.items()}


@pytest.fixture(scope="session")
def tokens(W):
    return [np.asarray(w[:256], dtype=np.int64) for w in W]
