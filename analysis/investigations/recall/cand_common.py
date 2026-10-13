"""Shared code for the recall_* candidates: RECALL with a new attention rule (fitted to head 1's own pattern by
fit_attn.py) and optional extra vote directions. The 15 RECALL vote directions (copy / topic / next x 5 word groups)
are recomputed with the new attention and replace the base head1.* directions."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import fit_attn  # noqa: E402
import lib  # noqa: E402
from program import head1  # noqa: E402

BASE_KEYS = [f"head1.{a}.{g}" for g in head1.GROUP_NAMES for a in ("copy", "topic", "next")]


def attention(P, t, variant):
    th = json.load(open(f"{lib.OUT}/fit_attn.json"))[variant]["theta"]
    names = fit_attn.VARIANTS[variant]
    X = fit_attn.features(P, t, names)
    score = X @ np.array([th[k] for k in names], np.float32)
    from program.readers import softmax_rows
    return softmax_rows(score.astype(np.float64))


def recall_features(P, t, variant):
    A = attention(P, t, variant)
    return A, lib.group_votes(P, t, A)


_TAB = {}


def family_table(P):
    """Cosine similarity of the topic profiles of two tokens (closed form from the program's topic table):
    rows of max(topic, 0) weighted by sqrt(P(v)), normalised, multiplied. she ~ her ~ herself ~ female names."""
    if "family" not in _TAB:
        s = P.STATS
        tp = np.maximum(np.asarray(s.topic, np.float64), 0.0) * np.sqrt(np.asarray(s.p_uni, np.float64))[None, :]
        tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-12
        _TAB["family"] = (tp @ tp.T).astype(np.float32)
    return _TAB["family"]


def near_table(P):
    """Mean within-story skip-gram PMI at distances 2..8 (program STATS.pmi)."""
    if "near" not in _TAB:
        s = P.STATS
        _TAB["near"] = np.mean([np.asarray(s.pmi[d], np.float32) for d in range(2, 9)], axis=0)
    return _TAB["near"]


def b25_table(P):
    if "b25" not in _TAB:
        import os
        path = f"{lib.OUT}/band_25_64.npy"
        if not os.path.exists(path):           # screening fallback: the head-1 card's cache (same counting rule)
            path = "/Users/svi/Desktop/trlm/analysis/scratch/head1_card/PMI_25_64.npy"
        _TAB["b25"] = np.load(path).astype(np.float32)
    return _TAB["b25"]


def per_group(P, t, A, table, name):
    g = head1.GROUP[t]
    tab = np.asarray(table[t], np.float64)
    return {f"{name}.{gn}": (A * (g == k)[None, :]) @ tab for k, gn in enumerate(head1.GROUP_NAMES)}
