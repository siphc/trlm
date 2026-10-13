"""Candidate (head 3): inside a word, the word's first token votes for how words that start with it continue.

Rule: at a query that is a word piece (starts with a lower-case letter) and continues a word whose first token
(space + letter) is s at distance 1..3 (every token in between is a word piece),
  word[v] = log P_w(v | s) - log P(v), P_w: v sits 1..4 tokens after s inside the same word (TRAIN corpus counts,
            Dirichlet 20 toward P(v); investigations/head3/tables.py)
Zero elsewhere. In the model, word-piece queries put most of head 3's attention on the word's first token
(investigations/head3/NOTES.md: ' fr' 0.93 from 'r' in 'frustr', ' happ' 0.77 from 'in' in 'happin').
"""
import os

import numpy as np

NAME = "head3_onset"
HEADS = [3]
_T = {}


def _tab(name):
    if name not in _T:
        from analysis.fitlib import REPO
        _T[name] = np.load(os.path.join(REPO, "analysis", "out", "candidates", "head3", name + ".npy")).astype(np.float64)
    return _T[name]


def features(P, tokens):
    t = np.asarray(tokens)
    n = len(t)
    piece = np.asarray(P.STATS.piece)[t] > 0
    start = np.asarray(P.STATS.shape)[t] == 0
    out = np.zeros((n, 2000))
    for i in range(1, n):
        if not piece[i]:
            continue
        j = i - 1
        while j >= 0 and i - j <= 3 and piece[j]:
            j -= 1
        if j >= 0 and i - j <= 3 and start[j]:
            out[i] = _tab("word_pmi")[t[j]]
    return {"word": out}
