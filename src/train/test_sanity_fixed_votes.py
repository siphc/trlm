"""Sanity tests for study A's fixed-vote models (src/train/fixed_votes.py; pre-registration logs/prereg_fixed_votes.md).
Run before any long run: .venv/bin/python -m pytest -q src/train/test_sanity_fixed_votes.py

The model tests use a small synthetic tables file; the training tests (resume, load, single-batch fit) use the real
tables in data/fixed_votes/tables.npz and are skipped until it is built.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.train import fixed_votes as FV
from src.train.train import TrainConfig, load_model, train

V = 2000
REAL = os.path.exists(FV.TABLES)


@pytest.fixture(scope="module")
def fake_tables(tmp_path_factory):
    rng = np.random.default_rng(0)
    arrs = {n: rng.normal(0, 1, (V, V)).astype(np.float32) for n in FV.TABLE_NAMES}
    arrs["follow"] = np.log(rng.dirichlet(np.ones(V), V)).astype(np.float32)
    arrs["p_uni"] = np.full(V, 1 / V, np.float32)
    path = str(tmp_path_factory.mktemp("tables") / "tables.npz")
    np.savez(path, **arrs, sha=FV.tables_sha(arrs), rms="{}")
    return path


def _model(path, seed=0, tables="next,copy,topic,skip2"):
    torch.manual_seed(seed)
    m = FV.FixedVoteLM(FV.FixedVoteConfig(tables=tables), path)
    with torch.no_grad():                       # sharper attention than at initialisation, so the tests can see it
        m.qk.weight.mul_(20.0)
        m.beta.copy_(torch.tensor([0.7, -0.4, 0.3, 1.1]))
    return m.double()


def test_default_off():
    assert TrainConfig().fixed_tables == ""


def test_causality(fake_tables):
    m = _model(fake_tables)
    x = torch.randint(0, V, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, V, (2, 5))
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.equal(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_tables_are_fixed_buffers(fake_tables):
    m = _model(fake_tables)
    names = {n for n, _ in m.named_parameters()}
    assert names == {"wte.weight", "wpe.weight", "ln.weight", "qk.weight", "beta_follow", "beta"}
    assert not m.tables.requires_grad and not m.follow.requires_grad
    assert set(m.state_dict()) == names                        # buffers are not saved in checkpoints
    x = torch.randint(0, V, (2, 10))
    m(x).logsumexp(-1).sum().backward()
    assert m.tables.grad is None and m.qk.weight.grad is not None and m.beta.grad is not None


def test_matches_naive_reference(fake_tables):
    """Logits equal an explicit loop over heads, queries and keys (numpy, float64)."""
    m = _model(fake_tables)
    x = torch.randint(0, V, (1, 6))
    with torch.no_grad():
        got = m(x)[0].numpy()
        e = (m.wte.weight[x[0]] + m.wpe.weight[:6]).numpy()
    g = m.ln.weight.detach().numpy()
    e = (e - e.mean(1, keepdims=True)) / np.sqrt(e.var(1, keepdims=True) + 1e-5) * g
    W = m.qk.weight.detach().numpy()
    H, Dh = 4, 32
    ref = m.beta_follow.item() * m.follow.numpy()[x[0]]
    for h in range(H):
        q, k = e @ W[h * Dh:(h + 1) * Dh].T, e @ W[H * Dh + h * Dh:H * Dh + (h + 1) * Dh].T
        for i in range(6):
            s = np.array([q[i] @ k[j] / np.sqrt(Dh) for j in range(i + 1)])
            a = np.exp(s - s.max())
            a /= a.sum()
            ref[i] += m.beta[h].item() * sum(a[j] * m.tables[h].numpy()[x[0, j]] for j in range(i + 1))
    assert np.abs(got - ref).max() < 1e-9


def test_identity_attention_gives_self_votes(fake_tables):
    m = _model(fake_tables)
    x = torch.randint(0, V, (2, 9))
    eye = torch.eye(9, dtype=torch.float64).expand(2, 9, 9)
    with torch.no_grad():
        got = m(x, override={h: eye for h in range(4)})
        ref = m.beta_follow * m.follow[x] + sum(m.beta[h] * m.tables[h][x] for h in range(4))
    assert torch.allclose(got, ref, atol=1e-10)


def test_sha_is_checked(fake_tables, tmp_path):
    m = _model(fake_tables)
    with pytest.raises(ValueError):
        FV.FixedVoteLM(FV.FixedVoteConfig(tables_sha="0" * 16), fake_tables)
    assert m.cfg.tables_sha == str(np.load(fake_tables)["sha"])


# ----------------------------------------------------------------------------- training (real tables)

def _cfg(out, **kw):
    base = dict(variant="fixed", seed=3, budget_tokens=20 * 8 * 256, batch_windows=8, device="cpu",
                eval_every=10**9, log_every=5, out=str(out), fixed_tables="next,copy,topic,skip2",
                fixed_tables_path=FV.TABLES)
    base.update(kw)
    return TrainConfig(**base)


@pytest.mark.skipif(not REAL, reason="data/fixed_votes/tables.npz not built")
def test_resume_bitwise_and_load(tmp_path):
    from src.train.test_sanity import _train_until

    train(_cfg(tmp_path / "full"))
    cfg = _cfg(tmp_path / "part")
    _train_until(cfg, stop_at=10)
    train(cfg)
    a, b = load_model(str(tmp_path / "full" / "final.pt")), load_model(str(tmp_path / "part" / "final.pt"))
    assert isinstance(a, FV.FixedVoteLM)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    x = torch.randint(0, V, (2, 16))
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
    assert os.path.getsize(tmp_path / "full" / "final.pt") < 2_000_000      # tables are not in the checkpoint


@pytest.mark.skipif(not REAL, reason="data/fixed_votes/tables.npz not built")
def test_single_batch_fit():
    """On one repeated batch the loss falls well below the bigram's (attention learns to pick useful keys)."""
    from src.train.data import BatchSampler, load_stream

    torch.manual_seed(0)
    m = FV.FixedVoteLM(FV.FixedVoteConfig())
    x = torch.as_tensor(BatchSampler(load_stream("train"), 0, 4).batch(0)[:, :65].astype(np.int64))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    f = torch.nn.functional.cross_entropy
    with torch.no_grad():
        bigram = float(f(m.follow[x[:, :-1]].reshape(-1, V), x[:, 1:].reshape(-1)))
    for _ in range(150):
        loss = f(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < bigram - 0.5, (loss.item(), bigram)


@pytest.mark.skipif(not REAL, reason="data/fixed_votes/tables.npz not built")
def test_real_tables():
    from analysis.mech_eval import stats

    z = np.load(FV.TABLES)
    assert str(z["sha"]) == FV.tables_sha({k: z[k] for k in z.files})
    s = stats()
    pu = z["p_uni"].astype(np.float64)
    t = np.asarray(s.pmi[2], np.float64)
    t = t - (t @ pu)[:, None]
    assert np.allclose(z["skip2"], t / np.sqrt(pu @ ((t * t) @ pu)), atol=1e-5)
    assert np.array_equal(z["follow"], np.asarray(s.log_follow, np.float32))
    for n in FV.TABLE_NAMES:
        tab = z[n].astype(np.float64)
        assert np.abs(tab @ pu).max() < 1e-6 and abs(pu @ ((tab * tab) @ pu) - 1) < 1e-5
