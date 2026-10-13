"""Candidate WORD ONSET RECALL 2: word_onset_recall.py plus one named strength for copying at word starts.

Same attention as word_onset_recall (RECALL + B_WORD * start(t_i) * max(0, PMI_1(t_i, t_j)), B_WORD fitted to head 1's
own attention), same 15 RECALL directions, and one extra direction

    start(t_i) * sum_{j < i} a(i, j) [v = t_j],    start(q) = (1 - piece(q)) * cont(q)

i.e. at a word start the recalled tokens are copied with an extra strength COPY_AT_START (one scalar).
"""
import importlib.util
import os

import numpy as np

NAME = "word_onset_recall2"
HEADS = [1]
_spec = importlib.util.spec_from_file_location("wor", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "word_onset_recall.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
REPLACES = _base.REPLACES
attention = _base.attention


def features(P, tokens):
    s = P.STATS
    t = np.asarray(tokens, dtype=np.int64)
    n = len(t)
    out = _base.features(P, t)
    a = attention(t, s) * (1 - np.eye(n))
    start = np.asarray(s.cont)[t] * (1.0 - np.asarray(s.piece)[t])
    copy = np.zeros((n, 2000))
    np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), (a * start[:, None]).reshape(-1))
    out["copy_at_start"] = copy
    return out
