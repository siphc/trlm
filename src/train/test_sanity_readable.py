"""Sanity tests for readable training (side study, logs/readable_training.md; pre-registration "Verification").
Run before any long run: .venv/bin/python -m pytest -q src/train/test_sanity_readable.py

Most tests use a small synthetic pool and synthetic vote targets in a temporary folder; the tiny-pool overfit test
uses the real pool (data/readable/pool) and is skipped until that pool is complete.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.train import readable as R
from src.train.data import load_stream, window
from src.train.train import TrainConfig, load_model, train

V, T = 2000, 256
N_TR, N_HO, K_OUT, K_ATT = 64, 64, 8, (4, 6, 4)


def _softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """A synthetic pool (real token windows, random but valid targets) and synthetic vote targets."""
    d = tmp_path_factory.mktemp("pool")
    rng = np.random.default_rng(0)
    s = load_stream("train")
    toks = np.stack([np.asarray(window(s, i)) for i in range(N_TR + N_HO)]).astype(np.uint16)
    np.save(d / "tokens.npy", toks)
    lp = np.log(_softmax(rng.normal(0, 2, (N_TR + N_HO, T, V))))
    ids = np.argsort(-lp[:N_TR], -1)[..., :K_OUT]
    top = np.take_along_axis(lp[:N_TR], ids, -1)
    np.save(d / "out_ids.npy", ids.astype(np.uint16))
    np.save(d / "out_lp.npy", top.astype(np.float16))
    np.save(d / "out_tail.npy", np.clip(1 - np.exp(top).sum(-1), 0, 1).astype(np.float32))
    np.save(d / "heldout_lp.npy", lp[N_TR:].astype(np.float16))
    mask = np.tril(np.ones((T, T), bool))
    att = np.where(mask, _softmax(np.where(mask, rng.normal(0, 2, (N_TR + N_HO, 3, T, T)), -np.inf)), 0.0)
    for h, k in enumerate(K_ATT):
        a = att[:N_TR, h]
        keys = np.argsort(-a, -1)[..., :k]
        p = np.take_along_axis(a, keys, -1)
        bad = keys > np.arange(T)[:, None]
        keys = np.where(bad, np.arange(T)[:, None], keys)
        p = np.where(bad, 0.0, p)
        np.save(d / f"att{h}_keys.npy", keys.astype(np.uint8))
        np.save(d / f"att{h}_p.npy", p.astype(np.float16))
        np.save(d / f"att{h}_tail.npy", np.clip(1 - p.sum(-1), 0, 1).astype(np.float32))
    np.save(d / "heldout_att.npy", att[N_TR:].astype(np.float16))
    json.dump({"n_train": N_TR, "n_heldout": N_HO, "k_out": K_OUT, "k_att": list(K_ATT), "indices_sha": "fake",
               "program_sha": "fake", "complete": True}, open(d / "meta.json", "w"))
    colw = rng.random(V)
    colw /= colw.sum()
    tg = rng.normal(0, 1, (3, V, V))
    tg = tg - (tg @ colw)[..., None]
    tg = tg / np.sqrt((tg * tg) @ colw)[..., None]
    roww = rng.random((3, V))
    roww /= roww.sum(1, keepdims=True)
    tpath = str(d / "targets.npz")
    np.savez(tpath, targets=tg.astype(np.float32), colw=colw.astype(np.float32), roww=roww.astype(np.float32),
             sha="fake")
    return str(d), tpath


def _cfg(out, fake, **kw):
    pool, targets = fake
    base = dict(variant="attn", seed=3, budget_tokens=20 * 8 * 256, batch_windows=8, device="cpu",
                eval_every=10**9, log_every=5, out=str(out), readable_pool=pool, readable_targets=targets)
    base.update(kw)
    return TrainConfig(**base)


def _same_weights(a, b):
    ma, mb = load_model(os.path.join(a, "final.pt")), load_model(os.path.join(b, "final.pt"))
    return all(torch.equal(p, q) for p, q in zip(ma.state_dict().values(), mb.state_dict().values()))


def test_default_off():
    assert TrainConfig().readable_arm == ""


def test_lambda_zero_is_bitwise_identical(fake, tmp_path):
    train(_cfg(tmp_path / "plain", fake))
    train(_cfg(tmp_path / "O0", fake, readable_arm="O", lambda_out=0.0))
    train(_cfg(tmp_path / "I0", fake, readable_arm="I", lambda_att=0.0, lambda_vote=0.0))
    assert _same_weights(tmp_path / "plain", tmp_path / "O0")
    assert _same_weights(tmp_path / "plain", tmp_path / "I0")
    rows = [json.loads(l) for l in open(tmp_path / "O0" / "log.jsonl")]
    assert "aux_out" in rows[-1] and "m_kl_model_prog" in rows[-1]       # the final step logs the metrics


def test_pool_sampler():
    a, b = R.PoolSampler(100, 3, 16), R.PoolSampler(100, 3, 16)
    for step in (0, 5, 6, 7, 40):
        assert np.array_equal(a.indices(step), b.indices(step))
    epoch = np.concatenate([a.indices(i) for i in range(25)])               # 25 x 16 = 4 epochs of 100
    assert (np.bincount(epoch, minlength=100) == 4).all()
    assert epoch.max() < 100
    assert not np.array_equal(R.PoolSampler(100, 4, 16).indices(0), a.indices(0))


@pytest.mark.parametrize("arm", ["O", "I"])
def test_resume_bitwise(fake, tmp_path, arm):
    from src.train.test_sanity import _train_until

    kw = {"O": dict(readable_arm="O", lambda_out=1.0),
          "I": dict(readable_arm="I", lambda_att=1.0, lambda_vote=0.5)}[arm]
    train(_cfg(tmp_path / "full", fake, **kw))
    cfg = _cfg(tmp_path / "part", fake, **kw)
    _train_until(cfg, stop_at=10)
    train(cfg)
    assert _same_weights(tmp_path / "full", tmp_path / "part")
    assert json.load(open(tmp_path / "part" / "readable.json"))["arm"] == arm


def test_topk_kl_reference_and_gradient():
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.normal(0, 2, (2, 3, 50)), dtype=torch.float64, requires_grad=True)
    p_full = _softmax(rng.normal(0, 2, (2, 3, 50)))
    k = 7
    ids = np.argsort(-p_full, -1)[..., :k]
    p_top = np.take_along_axis(p_full, ids, -1)
    tail = 1 - p_top.sum(-1)
    got = R.topk_forward_kl(logits, torch.as_tensor(ids), torch.as_tensor(np.log(p_top)), torch.as_tensor(tail))
    q = _softmax(logits.detach().numpy())
    q_top = np.take_along_axis(q, ids, -1)
    q_rest = 1 - q_top.sum(-1)
    ref = ((p_top * np.log(p_top / q_top)).sum(-1) + tail * np.log(tail / q_rest)).mean()
    full = (p_full * np.log(p_full / q)).sum(-1).mean()
    assert abs(float(got) - ref) < 1e-10
    assert float(got) <= full + 1e-12
    f = lambda z: R.topk_forward_kl(z, torch.as_tensor(ids), torch.as_tensor(np.log(p_top)), torch.as_tensor(tail))  # noqa: E731
    assert torch.autograd.gradcheck(f, (logits,))


def test_attention_loss_reference_and_head3_untouched(fake):
    pool = R.Pool(fake[0])
    b = pool.train_batch(np.arange(4), "cpu", attention=True)
    m = load_model("models/final/attn_s0/final.pt").train()
    for p in m.parameters():
        p.requires_grad_(True)
    _, pats = m(b["x"], return_patterns=True)
    loss = torch.stack([R.attention_topk_kl(pats[0][:, h], *b["att"][i]) for i, h in enumerate(R.HEADS)]).mean()
    loss.backward()
    W = m.blocks[0].attn.qkv.weight.grad
    D, Dh = 128, 32
    for block in (0, 1, 2):                                                 # q, k, v rows of head 3
        assert float(W[block * D + 3 * Dh: block * D + 4 * Dh].abs().max()) == 0.0
    assert float(W[:3 * Dh].abs().max()) > 0                                 # heads 0-2 q rows do get gradient
    # numpy reference for one head and one window
    keys, p, tail = (x[0].numpy() for x in b["att"][0])
    a = pats[0][0, 0].detach().numpy().astype(np.float64)
    p = p.astype(np.float64); tail = tail.astype(np.float64)
    s = p.sum(-1) + tail
    p, tail = p / s[:, None], tail / s
    mt = np.take_along_axis(a, keys, -1)
    valid = p > 0
    kl = np.where(valid, p * np.log(np.where(valid, p, 1) / np.maximum(mt, 1e-30)), 0).sum(-1)
    rest = np.maximum(1 - np.where(valid, mt, 0).sum(-1), 1e-30)
    kl += np.where(tail > 0, tail * np.log(np.maximum(tail, 1e-30) / rest), 0)
    got = R.attention_topk_kl(pats[0][:1, 0], *(x[:1] for x in b["att"][0]))
    assert abs(float(got) - kl[1:].mean()) < 1e-4


def test_ov_tables_match_analysis():
    from analysis.look_write import common as LW

    m = load_model("models/final/attn_s0/final.pt")
    ref = LW.ov_tables(LW.weights(m), 1.0)
    with torch.no_grad():
        got = R.ov_tables_torch(m).numpy()
    for i, h in enumerate(R.HEADS):
        rel = np.abs(got[i] - ref[h]).max() / np.abs(ref[h]).max()
        assert rel < 1e-4, (h, rel)


REAL_POOL = os.path.join("data", "readable", "pool", "meta.json")


@pytest.mark.skipif(not (os.path.exists(REAL_POOL) and json.load(open(REAL_POOL)).get("complete")),
                    reason="real pool not built yet")
@pytest.mark.parametrize("arm", ["O", "I"])
def test_tiny_pool_overfit(tmp_path, arm):
    kw = {"O": dict(readable_arm="O", lambda_out=10.0), "I": dict(readable_arm="I", lambda_att=10.0,
                                                                  lambda_vote=10.0)}[arm]
    cfg = TrainConfig(variant="attn", seed=3, budget_tokens=60 * 8 * 256, batch_windows=8, device="cpu",
                      eval_every=10**9, log_every=5, out=str(tmp_path / arm), readable_pool_limit=4, aux_windows=4,
                      peak_lr=2e-3, **kw)
    train(cfg)
    rows = [json.loads(l) for l in open(tmp_path / arm / "log.jsonl")]
    key = "aux_out" if arm == "O" else "aux_att"
    first, last = rows[0][key], rows[-1][key]
    assert last < 0.7 * first, (first, last)
