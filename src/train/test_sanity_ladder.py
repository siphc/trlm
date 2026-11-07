"""Sanity tests for the ladder of tiny models (logs/ladder_accounts/design.md, Tools). Run before any ladder training:

    .venv/bin/python -m pytest src/train/test_sanity_ladder.py -q

Needs data/char/ (python -m src.train.data_char prepare) and, for the sampler test, data/tokens/train.bin.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from src.train import data_char
from src.train.data import BatchSampler, load_stream, window
from src.train.ladder_runs import ARMS, RUNGS, config
from src.train.model import TinyLM
from src.train.train import build_model, ladder_model_config, train

pytestmark = pytest.mark.skipif(not os.path.exists("data/char/alphabet.json"), reason="char data not prepared")


def model(rung: str, arm: str, seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return build_model(config(rung, arm, seed, 1e-3, 1, "x", "cpu")).eval()


def test_alphabet_counted_on_the_train_stream_only():
    """Every kept character's train count equals its count in the train stream, so the alphabet was counted on exactly
    the text of the train stream (train minus dev); the dev stream is separate and the test split is never read."""
    alpha = data_char.load_alphabet()
    s = data_char.load_stream("train")
    counts = np.zeros(256, np.int64)
    for i in range(0, len(s), 200_000_000):
        counts += np.bincount(np.asarray(s[i:i + 200_000_000]), minlength=256)
    assert counts[2:alpha["vocab_size"]].tolist() == alpha["counts"]
    assert counts[1] == alpha["unk_count"]
    assert counts[alpha["vocab_size"]:].sum() == 0
    meta = json.load(open("data/char/meta.json"))
    assert set(meta["n_symbols"]) == {"train", "dev"}
    assert counts[0] == 2_099_719                       # one end-of-story symbol per train-minus-dev story


def test_eval_windows_come_from_dev():
    dev = data_char.load_stream("dev")
    for ctx in data_char.CONTEXTS:
        w = data_char.load_eval("dev_full", ctx)
        idx = np.load(f"data/char/eval/dev_full_c{ctx}_window_idx.npy")
        assert w.shape == (2000, ctx + 1) and len(set(idx.tolist())) == 2000
        for j in (0, 777, 1999):
            assert np.array_equal(w[j], dev[idx[j] * ctx: idx[j] * ctx + ctx + 1])
        assert np.array_equal(data_char.load_eval("dev_fast", ctx), w[:200])


@pytest.mark.skipif(not os.path.exists("data/tokens/train.bin"), reason="main-track stream missing")
def test_sampler_default_unchanged_for_the_main_track():
    s = load_stream("train")
    b = BatchSampler(s, 3, 4)
    got = b.batch(17)
    want = np.stack([np.asarray(window(s, int(i)), dtype=np.int64) for i in b.indices(17)])
    assert np.array_equal(got, want)


def test_char_sampler_windows():
    s = data_char.load_stream("train")
    b = BatchSampler(s, 0, 8, 64)
    x = b.batch(5)
    i = int(b.indices(5)[0])
    assert x.shape == (8, 65) and np.array_equal(x[0], s[64 * i: 64 * i + 65])


def test_baselines_normalised():
    t = data_char.load_baselines()
    for k, v in t.items():
        assert np.allclose(np.exp(v).sum(-1), 1.0), k


def test_noln_arm_is_the_identity():
    m = model("L3", "noln")
    assert not any(isinstance(x, torch.nn.LayerNorm) for x in m.modules())
    names = [n for n, _ in m.named_parameters()]
    assert names == ["wte.weight", "wpe.weight", "blocks.0.attn.qkv.weight", "blocks.0.attn.out.weight"]
    x = torch.randint(0, m.cfg.vocab_size, (2, 64))
    with torch.no_grad():
        h = m.wte(x) + m.wpe(torch.arange(64))[None]
        h = h + m.blocks[0].attn(h)[0]
        assert torch.allclose(m(x), h @ m.wte.weight.T, atol=1e-6)


@pytest.mark.parametrize("rung", ["L1", "L3", "L8"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_causality(rung, arm):
    m = model(rung, arm, seed=1)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)                              # large weights, so a leak would show
    T = RUNGS[rung]["context_length"]
    x = torch.randint(0, m.cfg.vocab_size, (3, T))
    t = T // 2
    y = x.clone()
    y[:, t:] = torch.randint(0, m.cfg.vocab_size, (3, T - t))
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.equal(a[:, :t], b[:, :t])
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_param_counts_by_hand():
    V = data_char.vocab_size()
    for rung, r in RUNGS.items():
        for arm, ln in ARMS.items():
            d, ctx = r["d_model"], r["context_length"]
            want = V * d + ctx * d + 4 * d * d + (2 * d if ln else 0)    # wte, wpe, qkv + out, ln1 + ln_f
            got = sum(p.numel() for p in TinyLM(ladder_model_config(config(rung, arm, 0, 1e-3, 1, "x", "cpu"))).parameters())
            assert got == want, (rung, arm, got, want)


def test_short_run_learns():
    """A short L5 run (8M symbols, CPU) beats the character bigram on dev_full; a short L1 run beats the unigram.
    Whether L1 beats the bigram is a result, recorded by the pilot, not a sanity condition."""
    meta = json.load(open("data/char/meta.json"))["baseline_ce_dev_full"]
    with tempfile.TemporaryDirectory() as d:
        r5 = train(config("L5", "ln", 0, 1e-2, 8_000_000, f"{d}/l5", "cpu"))
        r1 = train(config("L1", "ln", 0, 1e-2, 4_000_000, f"{d}/l1", "cpu"))
    out = {"L5_8M_dev_full": r5["dev_full_loss"], "L1_4M_dev_full": r1["dev_full_loss"], "baselines": meta}
    os.makedirs("logs/runs", exist_ok=True)
    with open("logs/runs/sanity_ladder_short_run.json", "w") as f:
        json.dump(out, f, indent=1)
    assert r5["dev_full_loss"] < meta["bigram_c64"]
    assert r1["dev_full_loss"] < meta["unigram_c64"]
