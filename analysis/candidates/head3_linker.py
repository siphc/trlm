"""Candidate (head 3): after a linking word, the word before it votes for what follows it two places later and for
words that can stand where it stands (parallel structure: "Mom and" -> Dad, "big," -> scary, "lot of" -> fun).

Rule: at a query in the ATTENTION class LINKERS, with s = the token just before it,
  pmi2[v]  = PMI_2(s, v)                               (program skip-gram table, distance 2)
  subst[v] = log sum_c P(c just before s) P1(v | c) - log P(v)   (TRAIN corpus; investigations/head3/tables.py)
Zero elsewhere. Head 3 puts 0.31 of a linker's attention on the previous token (circuit card).
"""
import os

import numpy as np

NAME = "head3_linker"
HEADS = [3]
_T = {}


def _tab(name):
    if name not in _T:
        from analysis.fitlib import REPO
        _T[name] = np.load(os.path.join(REPO, "analysis", "out", "candidates", "head3", name + ".npy")).astype(np.float64)
    return _T[name]


def _linker():
    if "linker" not in _T:
        from program.categories import ATTENTION_CLASSES
        from program.vocab import ID
        m = np.zeros(2000)
        m[[ID[x] for x in ATTENTION_CLASSES["LINKERS"]]] = 1.0
        _T["linker"] = m
    return _T["linker"]


def features(P, tokens):
    t = np.asarray(tokens)
    n = len(t)
    q = _linker()[t].copy()
    q[0] = 0.0
    prev = np.concatenate([[0], t[:-1]])
    pmi2 = np.asarray(P.STATS.pmi[2], np.float64)[prev] * q[:, None]
    sub = _tab("sub_left")[prev] * q[:, None]
    return {"pmi2": pmi2, "subst": sub}
