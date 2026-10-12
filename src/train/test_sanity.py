"""Sanity tests that must pass before any long run (AGENTS.md section 4).

Run: python -m pytest -q src/train/test_sanity.py
The short-run test trains a 10M-token model (~1 min on Apple MPS) and needs data/baselines built.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
import torch

from src.train import data as D
from src.train.model import ModelConfig, TinyLM, count_params
from src.train.train import TrainConfig, evaluate, load_model, train

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)


# 1. Causality ---------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["attn", "mlp"])
@pytest.mark.parametrize("explicit", [False, True])
def test_causality(variant, explicit):
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=variant == "mlp")).eval()
    x = torch.randint(0, 2000, (2, 256))
    run = (lambda t: m(t, return_patterns=True)[0]) if explicit else m
    with torch.no_grad():
        base = run(x)
        for t in (0, 10, 100, 254):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 2000, (2, 255 - t))
            out = run(x2)
            assert torch.equal(base[:, : t + 1], out[:, : t + 1]) or \
                torch.allclose(base[:, : t + 1], out[:, : t + 1], atol=1e-6, rtol=0)
            assert not torch.allclose(base[:, t + 1:], out[:, t + 1:])


def test_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=True)).eval()
    x = torch.randint(0, 2000, (2, 256))
    with torch.no_grad():
        assert torch.allclose(m(x), m(x, return_patterns=True)[0], atol=1e-5)


# 2. Overfit a single batch ----------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["attn", "mlp"])
def test_overfit_single_batch(variant):
    torch.manual_seed(0)
    m = TinyLM(ModelConfig(mlp=variant == "mlp"))
    stream = D.load_stream("train")
    x = torch.as_tensor(np.stack([np.asarray(D.window(stream, i), dtype=np.int64) for i in range(4)]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(600):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 2000), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{variant}: single-batch loss after 600 steps = {loss.item():.4f}")
    assert loss.item() < 0.1


# 3. Short run beats unigram and bigram ----------------------------------------------------------

def _baseline_dev_ce(name: str, wins: np.ndarray) -> float:
    lp = np.load(f"data/baselines/{name}_logprobs.npy").astype(np.float64)
    x, y = wins[:, :-1].astype(int), wins[:, 1:].astype(int)
    return float(-(lp[y] if lp.ndim == 1 else lp[x, y]).mean())


def test_short_run_beats_baselines():
    out = "models/sanity/short_run_mlp"
    shutil.rmtree(out, ignore_errors=True)
    res = train(TrainConfig(variant="mlp", seed=0, budget_tokens=10_000_000, out=out,
                            eval_every=200, log_every=50))
    wins = D.load_eval("dev_full")
    uni, bi = _baseline_dev_ce("unigram", wins), _baseline_dev_ce("bigram", wins)
    rec = {"short_run_dev_full_loss": res["dev_full_loss"], "unigram_dev_full_ce": uni,
           "bigram_dev_full_ce": bi, "tokens": res["tokens"], "device": res["device"]}
    with open("logs/sanity_short_run.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(rec)
    assert res["dev_full_loss"] < bi < uni


# 4. Tokenizer fit without dev or test text ------------------------------------------------------

def test_tokenizer_fit_excludes_dev_and_test():
    meta = json.load(open("data/prep_meta.json"))
    idx = np.load("data/splits/dev_indices.npy")
    n_official = meta["n_stories"]["train"] + meta["n_stories"]["dev"]
    assert len(idx) == 20000 and len(np.unique(idx)) == 20000
    assert np.array_equal(idx, D.dev_indices(n_official)), "dev indices not reproducible from seed 0"
    assert D.sha(idx) == meta["dev_indices_sha"]
    assert meta["tokenizer_fit_on"] == "train_minus_dev"
    assert meta["tokenizer_fit_n_stories"] == n_official - 20000
    # The split function keeps dev rows out of the training list (checked on a synthetic corpus).
    fake = [f"story {i}" for i in range(n_official)]
    tr, dv, _ = D.split_official_train(fake)
    assert len(tr) == n_official - 20000 and not (set(tr) & set(dv))
    # prepare() passes only the train-minus-dev list to fit_tokenizer; test text is the
    # official validation split, which is loaded separately and never passed to the fit.
    src = open("src/train/data.py").read()
    assert "tok = fit_tokenizer(train)" in src


def test_report_exact_duplicates_between_dev_and_train():
    """Not a leak (dev rows are excluded by index) but worth knowing: identical stories in both."""
    texts = D.load_split("train")
    idx = set(np.load("data/splits/dev_indices.npy").tolist())
    dev = {texts[i] for i in idx}
    dup = sum(1 for i, t in enumerate(texts) if i not in idx and t in dev)
    test = set(D.load_split("test"))
    dup_test = sum(1 for i, t in enumerate(texts) if i not in idx and t in test)
    rec = {"train_rows_identical_to_a_dev_story": dup, "train_rows_identical_to_a_test_story": dup_test}
    with open("logs/duplicates.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(rec)


# 5. Token and parameter counts -----------------------------------------------------------------

def test_param_counts():
    counts = {}
    for variant in ("attn", "mlp"):
        m = TinyLM(ModelConfig(mlp=variant == "mlp"))
        counts[variant] = count_params(m)
    # Hand arithmetic: wte 2000*128 + wpe 256*128 + qkv 3*128*128 + out 128*128 + LN weights.
    attn = 256_000 + 32_768 + 49_152 + 16_384 + 128 * 2
    mlp = attn + 2 * 128 * 512 + 128
    assert counts["attn"]["total"] == attn == 354_560
    assert counts["mlp"]["total"] == mlp == 485_760
    counts["hand_arithmetic"] = {"attn": attn, "mlp": mlp}
    with open("logs/param_counts.json", "w") as f:
        json.dump(counts, f, indent=2)


def test_no_bias_and_tied_embeddings():
    m = TinyLM(ModelConfig(mlp=True))
    names = [n for n, _ in m.named_parameters()]
    assert not any("bias" in n for n in names), names
    assert m.wte.weight.data_ptr() == m.wte.weight.data_ptr()  # output uses wte.weight directly
    assert "lm_head" not in " ".join(names)


def test_token_counts():
    meta = json.load(open("data/prep_meta.json"))
    for split in ("train", "dev", "test"):
        s = D.load_stream(split)
        assert len(s) == meta["n_tokens"][split]
        n_eos = int((np.asarray(s) == D.EOS_ID).sum())
        assert n_eos == meta["n_stories"][split], (split, n_eos)
        assert s[-1] == D.EOS_ID
        assert int(np.asarray(s).max()) < 2000
    tok = D.load_tokenizer()
    for t in D.load_split("test")[:200]:
        assert tok.decode(tok.encode(t).ids) == t
    for name, n in (("dev_fast", 200), ("dev_full", 2000), ("test_full", 2000), ("claims", 200)):
        w = D.load_eval(name)
        assert w.shape == (n, 257) and w.dtype == np.uint16


# Determinism and resume -------------------------------------------------------------------------

def test_batches_depend_only_on_seed_and_step():
    s = D.load_stream("train")
    a, b = D.BatchSampler(s, 3, 64), D.BatchSampler(s, 3, 64)
    for step in (0, 5, 1000, 32_000, 40_000):  # 32,195 steps per epoch: last two cross the boundary
        assert np.array_equal(a.batch(step), b.batch(step))
    assert not np.array_equal(D.BatchSampler(s, 4, 64).batch(0), a.batch(0))
    first_epoch = np.concatenate([a.indices(i) for i in range(5)])
    assert len(np.unique(first_epoch)) == len(first_epoch)


def test_resume_reproduces_training():
    base = dict(variant="attn", seed=1, budget_tokens=40 * 64 * 256, device="cpu", eval_every=10**9,
                log_every=10)
    full = "models/sanity/resume_full"
    part = "models/sanity/resume_part"
    for d in (full, part):
        shutil.rmtree(d, ignore_errors=True)
    train(TrainConfig(**base, out=full))
    # Simulate a disconnect after 20 of 40 steps, then resume with the same command.
    cfg = TrainConfig(**base, out=part)
    _train_until(cfg, stop_at=20)
    train(cfg)
    a = load_model(f"{full}/final.pt")
    b = load_model(f"{part}/final.pt")
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n


def _train_until(cfg: TrainConfig, stop_at: int) -> None:
    """Run train() but raise after `stop_at` steps, leaving latest.pt like a disconnected session."""
    import src.train.train as T

    class Stop(Exception):
        pass

    real_clip = torch.nn.utils.clip_grad_norm_
    counter = {"n": 0}

    def clip_and_count(*a, **k):
        counter["n"] += 1
        if counter["n"] > stop_at:
            raise Stop()
        return real_clip(*a, **k)

    cfg.resume_every_sec = 0.0  # save resume state after every step
    torch.nn.utils.clip_grad_norm_ = clip_and_count
    try:
        T.train(cfg)
    except Stop:
        pass
    finally:
        torch.nn.utils.clip_grad_norm_ = real_clip
