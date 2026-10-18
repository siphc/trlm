"""Sanity tests for study B's penalty against the look-follows-write law (src/train/law_pressure.py; pre-registration
logs/prereg_law_pressure.md). Run before any long run: .venv/bin/python -m pytest -q src/train/test_sanity_law_pressure.py
Needs data/fixed_votes/tables.npz (FOLLOW's bigram rule) and models/final/attn_s1.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from src.train import law_pressure as L
from src.train.model import ModelConfig, TinyLM
from src.train.train import TrainConfig, load_model, train

P1_PATH = "data/fixed_votes/tables.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(P1_PATH), reason="data/fixed_votes/tables.npz not built")


@pytest.fixture(scope="module")
def p1():
    return torch.as_tensor(np.exp(np.load(P1_PATH)["follow"].astype(np.float64)))


def _random_model(seed=0):
    torch.manual_seed(seed)
    m = TinyLM(ModelConfig(mlp=False)).double()
    with torch.no_grad():                                   # larger weights than at initialisation, so terms are visible
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(10.0)
    return m


def _batch(n=2, T=16, seed=0):
    from src.train.data import BatchSampler, load_stream

    return torch.as_tensor(BatchSampler(load_stream("train"), seed, n).batch(0)[:, :T + 1].astype(np.int64))


def test_default_off():
    c = TrainConfig()
    assert c.law_arm == "" and c.lambda_law == 0.0 and c.init_from == ""


def test_terms_match_reference(p1):
    """s equals the explicit attention score at (i, i - d); u equals OV_h(t_j, y_i) - sum_v P1(v | t_i) OV_h(t_j, v)
    with OV_h from src/train/readable.ov_tables_torch."""
    from analysis.look_write import common as C
    from src.train.readable import ov_tables_torch

    m = _random_model()
    x = _batch()
    T = x.shape[1] - 1
    with torch.no_grad():
        s, u = L.law_terms(m, x, p1)
        sc = C.attention_inputs(m, x[:, :T])[4]                  # (B, H, T, T)
        ov = ov_tables_torch(m, heads=range(4))                  # (H, V, V)
    for b in range(x.shape[0]):
        for qi, i in enumerate(range(L.D_MAX, T)):
            for d in range(1, L.D_MAX + 1):
                j = i - d
                tj, ti, y = x[b, j], x[b, i], x[b, i + 1]
                for h in range(4):
                    assert abs(float(s[h, b, qi, d - 1] - sc[b, h, i, j])) < 1e-9
                    ref = ov[h, tj, y] - (ov[h, tj] * p1[ti]).sum()
                    assert abs(float(u[h, b, qi, d - 1] - ref)) < 1e-8 * max(1.0, abs(float(ref)))


def test_double_centre():
    z = L.double_centre(torch.randn(4, 3, 5, L.D_MAX, dtype=torch.float64))
    assert z.mean(-1).abs().max() < 1e-12 and z.mean((1, 2)).abs().max() < 1e-12


@pytest.mark.parametrize("arm", L.ARMS)
def test_arm_gradients(p1, arm):
    m = _random_model()
    x = _batch()
    D = m.cfg.d_model

    class Cfg:
        law_arm, lambda_law, law_ramp, law_p1_path, total_steps = arm, 1.0, 0.0, P1_PATH, 1

    pen = L.LawPenalty(Cfg, "cpu")
    pen.p1 = p1
    loss, _ = pen.loss(m, x, 0)
    loss.backward()
    W = m.blocks[0].attn.qkv.weight.grad
    go = m.blocks[0].attn.out.weight.grad                     # None: the weight is not in the penalty's graph
    qk, v, o = W[:2 * D].abs().sum(), W[2 * D:].abs().sum(), 0.0 if go is None else go.abs().sum()
    if arm == "attn":
        assert qk > 0 and v == 0 and o == 0
    elif arm == "vote":
        assert qk == 0 and v > 0 and o > 0
    else:
        assert qk > 0 and v > 0 and o > 0


def _cfg(out, **kw):
    base = dict(variant="attn", seed=3, budget_tokens=20 * 8 * 256, batch_windows=8, device="cpu",
                eval_every=10**9, log_every=5, out=str(out))
    base.update(kw)
    return TrainConfig(**base)


def _same(a, b):
    ma, mb = load_model(os.path.join(a, "final.pt")), load_model(os.path.join(b, "final.pt"))
    return all(torch.equal(p, q) for p, q in zip(ma.state_dict().values(), mb.state_dict().values()))


def test_lambda_zero_is_bitwise_identical(tmp_path):
    train(_cfg(tmp_path / "plain"))
    train(_cfg(tmp_path / "law0", law_arm="both", lambda_law=0.0))
    assert _same(tmp_path / "plain", tmp_path / "law0")
    rows = [json.loads(l) for l in open(tmp_path / "law0" / "log.jsonl")]
    assert "law_rho_h0" in rows[-1]


def test_resume_bitwise(tmp_path):
    from src.train.test_sanity import _train_until

    kw = dict(law_arm="both", lambda_law=3.0, law_ramp=0.0)
    train(_cfg(tmp_path / "full", **kw))
    cfg = _cfg(tmp_path / "part", **kw)
    _train_until(cfg, stop_at=10)
    train(cfg)
    assert _same(tmp_path / "full", tmp_path / "part")
    assert json.load(open(tmp_path / "full" / "result.json"))["law_pressure"]["lambda"] == 3.0


def test_init_from(tmp_path):
    train(_cfg(tmp_path / "a"))
    train(_cfg(tmp_path / "b", init_from=str(tmp_path / "a" / "final.pt"), peak_lr=0.0, seed=4))
    assert _same(tmp_path / "a", tmp_path / "b")                 # learning rate 0: the loaded weights are kept


def test_trained_model_follows_the_law_and_penalty_can_remove_it(p1):
    """In attn_s1 the instance-level correlation is positive for most heads (the law, measured per token); optimising
    the penalty alone on one batch removes most of it."""
    m = load_model("models/final/attn_s1/final.pt")
    x = _batch(n=4, T=128, seed=1)
    p1f = p1.float()
    with torch.no_grad():
        rho0 = L.head_correlations(*L.law_terms(m, x, p1f))
    assert (rho0 > 0).sum() >= 3, rho0
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(60):
        rho = L.head_correlations(*L.law_terms(m, x, p1f))
        loss = (rho ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rho1 = L.head_correlations(*L.law_terms(m, x, p1f))
    assert float((rho1 ** 2).mean()) < 0.25 * float((rho0 ** 2).mean()), (rho0, rho1)
    assert all(math.isfinite(float(r)) for r in rho1)
