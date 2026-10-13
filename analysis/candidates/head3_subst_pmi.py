"""Candidate (head 3): substitutes (head3_subst) plus the distance-matched skip-gram vote of the card
(PMI_{d+1}(s, v), the earlier 'head-3 reader'), both through head 3's card attention rule, earlier tokens only."""
import os
import sys

import numpy as np

NAME = "head3_subst_pmi"
HEADS = [3]
_D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "investigations", "head3")
_T = {}


def _tab(name):
    if name not in _T:
        from analysis.fitlib import REPO
        _T[name] = np.load(os.path.join(REPO, "analysis", "out", "candidates", "head3", name + ".npy")).astype(np.float64)
    return _T[name]


def features(P, tokens):
    sys.path.insert(0, _D)
    import h3rule

    t = np.asarray(tokens)
    n = len(t)
    a = h3rule.attention(t) * (1.0 - np.eye(n))
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    pmi = np.zeros((n, 2000))
    for k in range(2, 9):
        m = (d == k - 1) if k < 8 else (d >= 7)
        pmi += (a * m) @ np.asarray(P.STATS.pmi[k], np.float64)[t]
    return {"subst": a @ _tab("sub_left")[t], "pmi": pmi}
