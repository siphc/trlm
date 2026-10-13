"""Candidate (head 3): the words head 3 reads vote for words that can STAND WHERE THEY STAND (substitutes).

Hypothesis (analysis/investigations/head3/NOTES.md): head 3 carries a phrase across a linking word by voting for
words of the same kind as the word it reads ("Mom and" -> Dad, "Lily and" -> Ben, "big," -> scary). Counting rule
(TRAIN corpus): sub(s, v) = log sum_c P(c just before s) P1(v | c) - log P(v): go back from s to the token before
it, then forward to what follows that token. Attention: head 3's rule from its circuit card (27 scalars fitted to
head 3's own pattern), earlier tokens only.
"""
import os
import sys

import numpy as np

NAME = "head3_subst"
HEADS = [3]
_D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "investigations", "head3")
_T = {}


def _tab(name):
    if name not in _T:
        from analysis.fitlib import REPO
        _T[name] = np.load(os.path.join(REPO, "analysis", "out", "candidates", "head3", name + ".npy")).astype(np.float64)
    return _T[name]


def prepare(stats, corpus):
    sys.path.insert(0, _D)


def features(P, tokens):
    sys.path.insert(0, _D)
    import h3rule

    t = np.asarray(tokens)
    a = h3rule.attention(t) * (1.0 - np.eye(len(t)))
    return {"subst": a @ _tab("sub_left")[t]}
