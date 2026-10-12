"""Baseline (not a replacement program): Witten-Bell trigram backing off to the train bigram.

P(c | a, b) = (n(a,b,c) + T(a,b) * P_bigram(c | b)) / (n(a,b) + T(a,b)), T = distinct continuations.
Row 0 has no two-token context and uses the bigram. Reads data/baselines at import.
"""
import os

import numpy as np

_D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "baselines")
_BI = np.exp(np.load(os.path.join(_D, "bigram_logprobs.npy")).astype(np.float64))
_K = np.load(os.path.join(_D, "trigram_keys.npy"))
_C = np.load(os.path.join(_D, "trigram_counts.npy"))
_CK = np.load(os.path.join(_D, "trigram_ctx_keys.npy"))
_CC = np.load(os.path.join(_D, "trigram_ctx_counts.npy"))
_CT = np.load(os.path.join(_D, "trigram_ctx_types.npy"))
V = 2000


def predict(tokens):
    t = np.asarray(tokens, dtype=np.int64)
    out = np.empty((len(t), V))
    out[0] = np.log(_BI[t[0]])
    for i in range(1, len(t)):
        a, b = t[i - 1], t[i]
        p = _BI[b].copy()
        ctx = a * V + b
        j = np.searchsorted(_CK, ctx)
        if j < len(_CK) and _CK[j] == ctx:
            n, T = _CC[j], _CT[j]
            lo, hi = np.searchsorted(_K, ctx * V), np.searchsorted(_K, ctx * V + V)
            counts = np.zeros(V)
            counts[_K[lo:hi] - ctx * V] = _C[lo:hi]
            p = (counts + T * p) / (n + T)
        out[i] = np.log(p)
    return out
