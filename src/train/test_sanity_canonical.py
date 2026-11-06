"""Sanity tests for study C part B's non-readable canonicaliser controls (logs/study_C_readable/prereg_canonical_addendum.md): arms O
and I with a frozen trained model (attn_s0) as the target, computed online (src/train/readable.py, readable_target).
Run before any part-B run: .venv/bin/python -m pytest -q src/train/test_sanity_canonical.py

Uses the synthetic pool of test_sanity_readable.py (real token windows) and the real attn_s0 checkpoint.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.train import readable as R
from src.train.runs import CONTROL_TARGET, readable_kw
from src.train.test_sanity_readable import _cfg, _same_weights, fake  # noqa: F401  (fake is a fixture)
from src.train.train import load_model, train

TEACHER = CONTROL_TARGET[len("model:"):]
CTL = {"Octl": dict(readable_arm="O", lambda_out=1.0, readable_target=CONTROL_TARGET),
       "Ictl": dict(readable_arm="I", lambda_att=1.0, lambda_vote=0.5, readable_target=CONTROL_TARGET)}


def test_run_settings():
    """The controls are arms O and I with only the target changed (same lambda scaling, c for the vote term)."""
    for arm in ("O", "I"):
        a, b = readable_kw(arm, 0.3, 0.84), readable_kw(arm + "ctl", 0.3, 0.84)
        assert b.pop("readable_target") == CONTROL_TARGET and a == b


def test_lambda_zero_is_bitwise_identical(fake, tmp_path):  # noqa: F811
    """Loading the frozen target model must not touch the random state: lambda 0 = the plain run, bitwise."""
    train(_cfg(tmp_path / "plain", fake))
    for arm in ("O", "I"):
        kw = dict(readable_arm=arm, readable_target=CONTROL_TARGET)
        train(_cfg(tmp_path / arm, fake, **kw))
        assert _same_weights(tmp_path / "plain", tmp_path / arm), arm
    rows = [json.loads(l) for l in open(tmp_path / "I" / "log.jsonl")]
    assert "m_kl_teacher_model" in rows[-1] and "m_att_kl_teacher_model" in rows[-1]


def test_resume_bitwise(fake, tmp_path):  # noqa: F811
    from src.train.test_sanity import _train_until

    for arm, kw in CTL.items():
        train(_cfg(tmp_path / f"{arm}_full", fake, **kw))
        cfg = _cfg(tmp_path / f"{arm}_part", fake, **kw)
        _train_until(cfg, stop_at=10)
        train(cfg)
        assert _same_weights(tmp_path / f"{arm}_full", tmp_path / f"{arm}_part"), arm
        s = json.load(open(tmp_path / f"{arm}_part" / "readable.json"))
        assert s["target"] == CONTROL_TARGET
        assert s["targets_sha"].startswith("model:") == (arm == "Ictl")


def test_target_model_scores_zero_against_itself(fake):  # noqa: F811
    """attn_s0 as the student: all three trained losses are 0 (up to float error); a different model is not."""
    aux = R.ReadableAux(_cfg("unused", fake, **CTL["Ictl"]), "cpu")
    m = load_model(TEACHER)
    other = load_model("models/final/attn_s1/final.pt")
    x = aux.pool.train_x(np.arange(4), "cpu")
    with torch.no_grad():
        ids, lp, tail = R.model_out_targets(aux.teacher, x, 64)
        assert float(R.topk_forward_kl(m(x), ids, lp, tail)) < 1e-5
        assert float(R.topk_forward_kl(other(x), ids, lp, tail)) > 0.05
        att = R.model_att_targets(aux.teacher, x, aux.k_att)
        for mod, small in ((m, True), (other, False)):
            _, pats = mod(x, return_patterns=True)
            l_att = torch.stack([R.attention_topk_kl(pats[0][:, h], *att[i]) for i, h in enumerate(R.HEADS)]).mean()
            assert (float(l_att) < 1e-5) == small, float(l_att)
        l_vote, rho = aux.vote_loss(m)
        assert float(l_vote) < 1e-5 and float(rho.min()) > 0.9999
        assert float(aux.vote_loss(other)[0]) > 0.1


def test_attention_targets_match_pool_format(fake):  # noqa: F811
    """Online attention targets follow build_pool.py: top-k keys per row, keys after the query point at the query
    with probability 0, tail = 1 - stored mass."""
    aux = R.ReadableAux(_cfg("unused", fake, **CTL["Ictl"]), "cpu")
    x = aux.pool.train_x(np.arange(2), "cpu")
    k = 6
    (keys, p, tail), = R.model_att_targets(aux.teacher, x, [k], heads=(1,))
    with torch.no_grad():
        a = aux.teacher(x, return_patterns=True)[1][0][:, 1].double().numpy()
    T = a.shape[-1]
    for w in range(2):
        for i in (0, 3, 100, 255):
            got = {}
            for j, v in zip(keys[w, i].tolist(), p[w, i].tolist()):   # padding repeats the query key with p = 0
                got[j] = got.get(j, 0.0) + v
            ref = np.argsort(-a[w, i])[:k]
            ref_keys = {int(j) for j in ref if j <= i}
            assert {j for j, v in got.items() if v > 0} == ref_keys, (w, i)
            assert all(j <= i for j in got)                                         # never a future key
            for j in ref_keys:
                assert abs(got[j] - a[w, i, j]) < 1e-5
            assert abs(float(tail[w, i]) - (1 - sum(a[w, i, j] for j in ref_keys))) < 1e-5
    assert T == 256


def test_vote_targets_standardised(fake):  # noqa: F811
    aux = R.ReadableAux(_cfg("unused", fake, **CTL["Ictl"]), "cpu")
    t, w = aux.targets.double(), aux.colw.double()
    assert float((t @ w).abs().max()) < 1e-4
    assert float(((t * t) @ w - 1).abs().max()) < 1e-3
