"""Shared helpers for the head-3 investigation (open track, analysis only).

Exact float64 numpy forward pass (analysis/scratch/head3_card/h3common.Net) plus head 3's earlier-token ablation
(the harness 'earlier' ablation: z_3 - a_ii v_i replaced by its mean over positions) and per-source logit votes.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "scratch", "head3_card"))

from h3common import Net, logsoftmax, kl_rows, class_ids, tok_strs  # noqa: E402

V, T, H3 = 2000, 256, 3
OUT = os.path.join(REPO, "analysis", "out", "candidates", "head3")


class H3Run:
    def __init__(self):
        self.net = Net()
        n = self.net
        self.Wv, self.WO = n.Wv[H3], n.WO[H3]

    def mean_ctx(self, windows):
        """Mean of head 3's earlier-token part (z - a_ii v_i) in the 32-dim head space, over all positions."""
        acc, cnt = np.zeros(self.net.Dh), 0
        for w in windows:
            r = self.net.run(w[:T])
            a, v = r["pats"][H3], r["vs"][H3]
            z = a @ v - np.diag(a)[:, None] * v
            acc += z.sum(0)
            cnt += len(z)
        return acc / cnt

    def run(self, w, mc):
        """Model log-probs, earlier-ablated log-probs, head-3 pattern, value vectors, final-LN scale."""
        n = self.net
        r = n.run(w[:T])
        a, v = r["pats"][H3], r["vs"][H3]
        z_self = np.diag(a)[:, None] * v
        out_abl = (z_self + mc[None, :]) @ self.WO.T
        lpa = logsoftmax(n.logits_with_out(r, out_abl))
        r["lpa"] = lpa
        r["a3"] = a
        r["v3"] = v
        return r

    def source_votes(self, r):
        """u_j: the logit vote (V,) of one unit of head-3 attention on source j, divided by the mean final-LN
        scale later. Returns (n, V) un-scaled by sdf (divide by sdf_i for the vote at query i)."""
        n = self.net
        o = r["v3"] @ self.WO.T
        o = (o - o.mean(-1, keepdims=True)) * n.gf
        return o @ n.WE.T


def wcorr(a, b, w):
    a = a - (w * a).sum(-1, keepdims=True)
    b = b - (w * b).sum(-1, keepdims=True)
    return (w * a * b).sum(-1) / np.sqrt((w * a * a).sum(-1) * (w * b * b).sum(-1) + 1e-30)
