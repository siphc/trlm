"""Shared helpers for the ov02 investigation (what heads 0 and 2 write from earlier tokens)."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "ov02")
V, D, DH = 2000, 128, 32


def weights():
    from analysis.tools import Model
    m = Model("attn_s0")
    m.WV = [m.Wqkv[2 * D + h * DH:2 * D + (h + 1) * DH] for h in range(4)]
    m.WQ = [m.Wqkv[h * DH:(h + 1) * DH] for h in range(4)]
    m.WK = [m.Wqkv[D + h * DH:D + (h + 1) * DH] for h in range(4)]
    m.WOh = [m.WO[:, h * DH:(h + 1) * DH] for h in range(4)]
    return m


def ln(x, g, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    sd = np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps)
    return xc / sd * g, sd[..., 0]


SDF_TYPICAL = 1.127   # typical final-LN scale (head-0 card)


def ov_tables(m, pos=None):
    """OV[h][s, v]: logits written by head h when it attends fully to token s (at a typical position),
    final LN linearised at the typical scale. Rows centred over v."""
    p = m.WP[1:].mean(0) if pos is None else m.WP[pos]
    h1, _ = ln(m.WE + p, m.g1)
    out = []
    for h in range(4):
        o = (h1 @ m.WV[h].T) @ m.WOh[h].T
        o = o - o.mean(-1, keepdims=True)
        lg = (o * m.gf / SDF_TYPICAL) @ m.WE.T
        out.append(lg - lg.mean(1, keepdims=True))
    return out


def ov_cached():
    path = os.path.join(OUT, "ov.npy")
    if not os.path.exists(path):
        np.save(path, np.stack(ov_tables(weights())).astype(np.float32))
    return np.load(path)


def rowcorr(A, B, wv=None):
    """Per-row correlation over v (weights wv over v, default uniform)."""
    if wv is None:
        wv = np.full(A.shape[1], 1.0 / A.shape[1])
    a = A - (A * wv).sum(1, keepdims=True)
    b = B - (B * wv).sum(1, keepdims=True)
    return (a * b * wv).sum(1) / np.sqrt((a * a * wv).sum(1) * (b * b * wv).sum(1) + 1e-30)


def double_centre(A, ws, wv):
    A = A - (A * wv).sum(1, keepdims=True)
    return A - (ws[:, None] * A).sum(0, keepdims=True)


def global_r2(y, Xs, ws, wv):
    """Weighted least squares of y[s, v] on the list of tables Xs (all double-centred); weights ws[s] wv[v]."""
    W = ws[:, None] * wv[None, :]
    y = double_centre(y, ws, wv)
    Xs = [double_centre(x, ws, wv) for x in Xs]
    k = len(Xs)
    G = np.array([[(W * Xs[i] * Xs[j]).sum() for j in range(k)] for i in range(k)])
    b = np.array([(W * Xs[i] * y).sum() for i in range(k)])
    c = np.linalg.solve(G, b)
    sst = (W * y * y).sum()
    return float(1 - (sst - 2 * c @ b + c @ G @ c) / sst), c


_M = None


def model_weights():
    global _M
    if _M is None:
        _M = weights()
    return _M


def model_attention(tokens):
    """Exact attention patterns of the model, (4, n, n)."""
    m = model_weights()
    t = np.asarray(tokens, dtype=np.int64)
    n = len(t)
    h1, _ = ln(m.WE[t] + m.WP[:n], m.g1)
    out = np.zeros((4, n, n))
    mask = np.tril(np.ones((n, n), dtype=bool))
    for h in range(4):
        q, k = h1 @ m.WQ[h].T, h1 @ m.WK[h].T
        s = np.where(mask, q @ k.T / np.sqrt(DH), -np.inf)
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        out[h] = e / e.sum(1, keepdims=True)
    return out
