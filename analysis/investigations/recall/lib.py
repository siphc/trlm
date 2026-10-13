"""Shared helpers for the RECALL investigation (head 1 of attn_s0).

Exact model internals (open track, insight only) come from analysis/scratch/head1_card/h1lib.Net:
  A1[i, j]   head 1's attention, ov[j] (V,) the logits a full attention weight on key j adds times sdf_i,
  sdf[i]     the final-LayerNorm scale. Exact head-1 logit contribution at i = sum_j A1[i, j] ov[j] / sdf[i].
Program side: program.head1 attention and per-group vote directions for any attention matrix.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "scratch", "head1_card"))
V = 2000
OUT = os.path.join(REPO, "analysis", "out", "candidates", "recall")


def net():
    import h1lib
    return h1lib.Net()


def group_votes(P, t, A, groups=None):
    """The 15 RECALL vote directions (copy/topic/next x 5 groups) for attention A (n, n), as a dict."""
    from program import head1
    s = P.STATS
    n = len(t)
    g = head1.GROUP[t] if groups is None else groups
    topic, nxt = np.asarray(s.topic[t], np.float64), np.asarray(s.pmi[1][t], np.float64)
    out = {}
    for k, gname in enumerate(head1.GROUP_NAMES):
        Ak = A * (g == k)[None, :]
        copy = np.zeros((n, V))
        np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), Ak.reshape(-1))
        out[f"copy.{gname}"] = copy
        out[f"topic.{gname}"] = Ak @ topic
        out[f"next.{gname}"] = Ak @ nxt
    return out


def copy_matrix(t, A):
    n = len(t)
    copy = np.zeros((n, V))
    np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), A.reshape(-1))
    return copy


class R2:
    """Pooled probability-weighted R^2 of y (the head's earlier-token effect) on sets of directions.
    Directions must be pre-centred over positions by the caller (see center)."""

    def __init__(self, sets, classes=("all",)):
        self.sets = sets
        self.acc = {(c, lab): [None, None, 0.0] for c in classes for lab in sets}
        self.classes = classes

    def add(self, w, y, feats, cls=None, chunk=32):
        y = y - (w * y).sum(1, keepdims=True)
        for c in self.classes:
            m = np.ones(len(w), bool) if c == "all" else (cls == c)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            for lab, ks in self.sets.items():
                a = self.acc[(c, lab)]
                for s in range(0, len(idx), chunk):
                    ii = idx[s : s + chunk]
                    wm, ym = w[ii], y[ii]
                    X = np.stack([feats[k][ii] for k in ks], -1)
                    X = X - (wm[..., None] * X).sum(1, keepdims=True)
                    xtx = np.einsum("nvk,nv,nvl->kl", X, wm, X, optimize=True)
                    xty = np.einsum("nvk,nv,nv->k", X, wm, ym, optimize=True)
                    a[0] = xtx if a[0] is None else a[0] + xtx
                    a[1] = xty if a[1] is None else a[1] + xty
                    a[2] += float((wm * ym * ym).sum())

    def result(self):
        out = {}
        for (c, lab), (xtx, xty, sst) in self.acc.items():
            if xtx is None or sst == 0:
                continue
            beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
            r2 = 1 - (sst - 2 * beta @ xty + beta @ xtx @ beta) / sst
            out.setdefault(lab, {})[c] = {"r2": round(float(r2), 4), "beta": [round(float(b), 3) for b in beta][:20]}
        return out


def position_classes(P, t):
    from analysis.head_effects import position_classes as pc
    return pc(P, t)
