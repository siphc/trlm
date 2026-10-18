"""Sanity tests for the multi-block side study (logs/decisions.md item 8), run before its long runs.

Run: python -m pytest -q src/train/test_sanity_deep.py
The main track's tests (test_sanity.py) are unchanged; these cover attention-only models with 2 and 4 blocks and
check that the n_blocks option leaves the 1-block main track exactly as it was.
"""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest
import torch

from src.train import data as D
from src.train.model import ModelConfig, TinyLM, count_params
from src.train.test_sanity import _baseline_dev_ce, _train_until
from src.train.train import TrainConfig, evaluate, load_model, model_config, train

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
DEPTHS = [2, 4]


@pytest.mark.parametrize("n_blocks", DEPTHS)
@pytest.mark.parametrize("explicit", [False, True])
def test_causality_deep(n_blocks, explicit):
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=False, n_blocks=n_blocks)).eval()
    x = torch.randint(0, 2000, (2, 256))
    run = (lambda t: m(t, return_patterns=True)[0]) if explicit else m
    with torch.no_grad():
        base = run(x)
        for t in (0, 10, 100, 254):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 2000, (2, 255 - t))
            out = run(x2)
            assert torch.allclose(base[:, : t + 1], out[:, : t + 1], atol=1e-6, rtol=0)
            assert not torch.allclose(base[:, t + 1:], out[:, t + 1:])


@pytest.mark.parametrize("n_blocks", DEPTHS)
def test_sdpa_matches_explicit_deep(n_blocks):
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=False, n_blocks=n_blocks)).eval()
    x = torch.randint(0, 2000, (2, 256))
    with torch.no_grad():
        logits, patterns = m(x, return_patterns=True)
        assert torch.allclose(m(x), logits, atol=1e-5)
    assert len(patterns) == n_blocks and all(p.shape == (2, 4, 256, 256) for p in patterns)


@pytest.mark.parametrize("n_blocks", DEPTHS)
def test_overfit_single_batch_deep(n_blocks):
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=False, n_blocks=n_blocks))
    stream = D.load_stream("train")
    x = torch.as_tensor(np.stack([np.asarray(D.window(stream, i), dtype=np.int64) for i in range(4)]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(600):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 2000), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{n_blocks} blocks: single-batch loss after 600 steps = {loss.item():.4f}")
    assert loss.item() < 0.1


def test_param_counts_and_init_deep():
    per_block = 3 * 128 * 128 + 128 * 128 + 128            # qkv + out + ln1 (attention-only block, no biases)
    base = 2000 * 128 + 256 * 128 + 128                     # wte + wpe + ln_f
    rec = {"per_block_hand_arithmetic": per_block}
    for n in [1] + DEPTHS:
        torch.manual_seed(0)
        m = TinyLM(ModelConfig(mlp=False, n_blocks=n))
        total = count_params(m)["total"]
        assert total == base + n * per_block
        expected_std = 0.02 / math.sqrt(2 * n)
        for blk in m.blocks:
            std = float(blk.attn.out.weight.detach().std())
            assert abs(std - expected_std) / expected_std < 0.05, (n, std, expected_std)
        rec[f"attn_{n}L"] = {"total": total, "out_proj_init_std": expected_std}
    assert rec["attn_1L"]["total"] == 354_560 and rec["attn_2L"]["total"] == 420_224
    assert rec["attn_4L"]["total"] == 551_552
    with open("logs/param_counts_deep.json", "w") as f:
        json.dump(rec, f, indent=2)


def test_main_track_unchanged():
    """n_blocks defaults to 1: the main track's config and its trained model behave exactly as before."""
    assert TrainConfig().n_blocks == 1
    assert model_config("attn") == ModelConfig(mlp=False) and model_config("mlp") == ModelConfig(mlp=True)
    res = json.load(open("models/final/attn_s0/result.json"))
    m = load_model("models/final/attn_s0/final.pt")
    loss = evaluate(m, D.load_eval("dev_full"), torch.device("cpu"))
    assert abs(loss - res["dev_full_loss"]) < 1e-4, (loss, res["dev_full_loss"])


def test_resume_reproduces_training_deep():
    base = dict(variant="attn", seed=1, n_blocks=2, budget_tokens=40 * 64 * 256, device="cpu", eval_every=10**9,
                log_every=10)
    full, part = "models/sanity/resume_full_2L", "models/sanity/resume_part_2L"
    for d in (full, part):
        shutil.rmtree(d, ignore_errors=True)
    train(TrainConfig(**base, out=full))
    cfg = TrainConfig(**base, out=part)
    _train_until(cfg, stop_at=20)
    train(cfg)
    a, b = load_model(f"{full}/final.pt"), load_model(f"{part}/final.pt")
    assert a.cfg.n_blocks == 2 and len(a.blocks) == 2
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n


@pytest.mark.parametrize("n_blocks", [1] + DEPTHS)
def test_short_run_beats_baselines_deep(n_blocks):
    """Attention-only models learn more slowly than the mlp variant used by test_sanity.py: at 10M tokens and LR 1e-3
    a 2-block model is still above the bigram (3.95 vs 3.52), and the 1-block attn_s0 run was above it at 8M tokens.
    So this test uses 25M tokens at the frozen peak LR, with the 1-block attention model as the reference."""
    out = f"models/sanity/short_run_attn{n_blocks}L"
    shutil.rmtree(out, ignore_errors=True)
    res = train(TrainConfig(variant="attn", n_blocks=n_blocks, seed=0, budget_tokens=25_000_000, peak_lr=2e-3,
                            out=out, eval_every=200, log_every=50))
    wins = D.load_eval("dev_full")
    uni, bi = _baseline_dev_ce("unigram", wins), _baseline_dev_ce("bigram", wins)
    path = "logs/runs/sanity_short_run_deep.json"
    rec = json.load(open(path)) if os.path.exists(path) else {}
    rec[f"attn_{n_blocks}L"] = {"short_run_dev_full_loss": res["dev_full_loss"], "unigram_dev_full_ce": uni,
                                "bigram_dev_full_ce": bi, "tokens": res["tokens"], "device": res["device"],
                                "seconds": res["elapsed"]}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(rec[f"attn_{n_blocks}L"])
    assert res["dev_full_loss"] < bi < uni
