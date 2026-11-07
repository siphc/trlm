"""Self-tests of the ladder tools on a pilot model (models/ladder_pilot/L1_ln_lr1e-2_b25M):

    .venv/bin/python -m pytest analysis/ladder/test_tools.py -q

- eval: an account that pastes the model's own weights into the reference forward pass gets KL ~ 0 (P1 passes);
  the character bigram as an account gets KL(model || bigram).
- scrub: a claim whose key is exactly what the site is computed from (direct path: symbol and position) scores
  faithfulness 1; a constant key scores about 0; a key unique to each position fails the donor rule.
- length: the weights reference is found and the pasted-weights account is not shorter than it by much.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np
import pytest

from analysis.ladder import common, eval as ev, length, scrub

PILOT = os.path.join(common.REPO, "models", "ladder_pilot", "L1_ln_lr1e-2_b25M")
pytestmark = pytest.mark.skipif(not os.path.exists(f"{PILOT}/final.pt"), reason="pilot model missing")


@pytest.fixture(autouse=True)
def pilot_as_seed0(monkeypatch):
    monkeypatch.setattr(common, "model_dir", lambda rung, arm, seed: PILOT)


def pasted_account(path: str) -> None:
    m = common.load("L1", "ln", 0)
    lines = ["import numpy as np", "W = {}"]
    for k, v in m.state_dict().items():
        lines.append(f"W[{k!r}] = np.array({v.numpy().astype(np.float64).tolist()!r})")
    with open(os.path.join(os.path.dirname(length.__file__), "reference_forward.py")) as f:
        ref = f.read()
    body = ref.split('"""', 2)[2].replace("import numpy as np\n", "")
    lines += [body, "def predict(X):\n    return forward(X, W, 1)"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def claims_account(path: str, head_key: str) -> None:
    src = f"""
    import numpy as np
    def predict(X):
        raise NotImplementedError
    def k_direct(X):
        return np.stack([X, np.broadcast_to(np.arange(X.shape[1]), X.shape)], -1)
    def k_head(X):
        n, T = X.shape
        return {head_key}
    CLAIMS = [{{"name": "direct", "site": "direct", "key": k_direct}},
              {{"name": "head", "site": "head0", "key": k_head}}]
    """
    with open(path, "w") as f:
        f.write(textwrap.dedent(src))


def test_eval_pasted_weights(tmp_path):
    p = str(tmp_path / "acc.py")
    pasted_account(p)
    assert length.rule_errors(p) == []
    r = ev.score("L1", "ln", 0, p, "dev_fast")
    assert r["kl"] < 1e-8 and r["P1"] and r["top1"] > 0.999


def test_scrub_exact_constant_unique(tmp_path):
    p = str(tmp_path / "c.py")
    claims_account(p, "np.zeros((n, T), int)")
    r = scrub.run("L1", "ln", 0, p, "dev_full")
    assert r["claims"]["direct"]["faithfulness"] > 0.9999 and r["claims"]["direct"]["pass"]
    assert r["claims"]["head"]["faithfulness"] < 0.1 and not r["P3"]
    claims_account(p, "np.arange(n * T).reshape(n, T)")
    r = scrub.run("L1", "ln", 0, p, "dev_fast")
    assert r["claims"]["head"]["no_donor_share"] == 1.0 and not r["claims"]["head"]["pass"]


def test_replacement_exact_and_zero(tmp_path):
    """A value claim that computes the head's output exactly (here from the model's weights, as a check of the
    plumbing) scores 1; one that returns zeros scores well below."""
    m = common.load("L1", "ln", 0).double()
    W = common.windows("L1", "dev_full")
    _, z = scrub.parts(m, W[:, :-1].astype(np.int64))
    p = str(tmp_path / "v.py")
    claims_account(p, "np.zeros((n, T), int)")
    acc = common.load_account(p)
    exact = {"name": "head", "site": "head0", "value": lambda X: z[:, :, 0]}
    zero = {"name": "head", "site": "head0", "value": lambda X: np.zeros_like(z[:, :, 0])}
    for c, ok in ((exact, True), (zero, False)):
        acc.CLAIMS[1] = c
        orig = common.load_account
        try:
            scrub.load_account = lambda path: acc
            r = scrub.run("L1", "ln", 0, p, "dev_full")
        finally:
            scrub.load_account = orig
        f = r["claims"]["head"]["faithfulness"]
        assert r["claims"]["head"]["test"] == "replace"
        assert (f > 0.9999) if ok else (f < 0.9), f


def test_donors_never_self_and_same_key():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50, 5000)
    d, has = scrub.donors(ids, rng)
    assert np.all(ids[d] == ids)
    counts = np.bincount(ids)
    assert np.all(has == (counts[ids] > 1))


def test_length_reference(tmp_path):
    r = length.weights_reference("L1", "ln", 0, "dev_fast")
    assert r["found"] and r["ladder"][-1]["kl"] <= 0.01
    p = str(tmp_path / "acc.py")
    pasted_account(p)
    assert length.source_bits(p) > 0.5 * r["total_bits"]
