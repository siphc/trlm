"""Self-tests of the ladder tools that remain after decision 23, on a pilot model
(models/ladder_pilot/L1_ln_lr1e-2_b25M):

    .venv/bin/python -m pytest analysis/ladder/test_tools.py -q

- parts() and finish() reproduce the model's log-probs; resampled() changes them for each site.
- reference_forward.forward reproduces the model from its weights.
- eval.score reports the baselines and a dev loss between the model's and the unigram's.
- cost.source_bits counts a proof file and grows with its content.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from analysis.ladder import common, eval as ev
from analysis.ladder.reference_forward import forward
from analysis.proofs import cost

PILOT = os.path.join(common.REPO, "models", "ladder_pilot", "L1_ln_lr1e-2_b25M")
pytestmark = pytest.mark.skipif(not os.path.exists(f"{PILOT}/final.pt"), reason="pilot model missing")


@pytest.fixture(autouse=True)
def pilot_as_seed0(monkeypatch):
    monkeypatch.setattr(common, "model_dir", lambda rung, arm, seed: PILOT)


def test_parts_finish_resampled():
    m = common.load("L1", "ln", 0).double()
    W = common.windows("L1", "dev_fast")[:50]
    X = W[:, :-1].astype(np.int64)
    x0, z = common.parts(m, X)
    lp = common.finish(m, x0, z)
    assert np.abs(lp - common.model_logprobs(m, W)).max() < 1e-4
    for site in ("direct", "head0"):
        lr = common.resampled(m, x0, z, site, np.random.default_rng(0))
        assert common.kl_pos(lp, lr).mean() > 1e-3


def test_reference_forward():
    m = common.load("L1", "ln", 0).double()
    W = common.windows("L1", "dev_fast")[:20]
    Wt = {k: v.numpy().astype(np.float64) for k, v in m.state_dict().items()}
    assert np.abs(forward(W[:, :-1].astype(np.int64), Wt, 1) - common.model_logprobs(m, W)).max() < 1e-4


def test_eval_score():
    r = ev.score("L1", "ln", 0, "dev_fast")
    assert r["ce_trigram"] < r["ce_bigram"] < r["ce_unigram"]
    assert r["ce_model"] < r["ce_unigram"]


def test_cost(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text('"""doc"""\nx = 1\n')
    b.write_text('"""doc"""\nx = 1\n' + "".join(f"y{i} = {i * 7919 % 1000}\n" for i in range(200)))
    assert 0 < cost.source_bits(str(a)) < cost.source_bits(str(b))
