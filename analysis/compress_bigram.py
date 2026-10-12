"""Compress the model's mean bigram (analysis/mean_dists.py) into readable tables.

Readable form, for current token t with word class c = class(t):
  q(v | t) = TOP[t][v]                                  if v is one of t's listed continuations
           = rest(t) * CLASS_NEXT[c][class(v)] * WITHIN[class(v)][v] / Z(t)   otherwise
where rest(t) = 1 - sum of listed probabilities and Z(t) renormalises the fallback over unlisted tokens.

For any predictor that depends only on t,
  KL(model || q) = KL(model || mean_bigram) + E_t KL(mean_bigram[t] || q[t]),
so the compression is scored on the mean bigram alone.

Library: build(mean, count, class_of, k_of) -> tables; score(...) -> excess KL.
"""
from __future__ import annotations

import numpy as np

V = 2000


def class_tables(big: np.ndarray, cnt: np.ndarray, uni: np.ndarray, lab: np.ndarray, K: int):
    """CLASS_NEXT[c_from, c_to] and WITHIN[v] (probability of v inside its own class)."""
    onehot = np.zeros((V, K))
    onehot[np.arange(V), lab] = 1.0
    to_class = big @ onehot                                   # (V, K) P(class of next | t)
    w = cnt[:, None] * to_class
    cn = np.zeros((K, K))
    np.add.at(cn, lab, w)
    cn = cn / np.maximum(cn.sum(1, keepdims=True), 1e-30)
    within = uni / np.maximum((uni @ onehot)[lab], 1e-30)
    return cn, within


def build(big, cnt, uni, lab, K, k_of, floor=1e-9):
    cn, within = class_tables(big, cnt, uni, lab, K)
    fallback = cn[lab][:, lab] * within[None, :]               # (V, V) unnormalised
    q = np.empty_like(big)
    tops = []
    for t in range(V):
        k = int(k_of[t])
        idx = np.argsort(-big[t])[:k] if k > 0 else np.array([], int)
        f = fallback[t].copy()
        f[idx] = 0.0
        f = np.maximum(f, floor)
        f[idx] = 0.0
        rest = 1.0 - big[t, idx].sum()
        q[t] = f / f.sum() * rest
        q[t, idx] = big[t, idx]
        tops.append(idx)
    return q, cn, within, tops


def excess_kl(big, cnt, q) -> float:
    w = cnt / cnt.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(big > 0, big * (np.log(big) - np.log(q)), 0.0).sum(1)
    return float((w * r).sum())
