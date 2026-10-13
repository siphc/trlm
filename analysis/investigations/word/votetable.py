"""With each head's REAL attention (insight only), which corpus vote table best describes what a read token writes?
Earlier-token directions sum_{j<i} a_ij T(t_j, .) for several closed-form tables T; pooled weighted R^2 of the head's
earlier-token effect, single tables and joint sets, by position class.

  python -m analysis.investigations.word.votetable [n_windows]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.decomp import parts
from analysis.investigations.word.diag import classes
from analysis.investigations.word.effects2 import pmi_table
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from harness.common import head_means, load_model
from program.readers import skipgram_votes

V = 2000
CLASSES = ["inside", "onset", "after multi-token word", "other", "all"]


def r2_set(ys, Xs, ws):
    K = Xs[0].shape[-1]
    xtx, xty, sst = np.zeros((K, K)), np.zeros(K), 0.0
    for y, X, w in zip(ys, Xs, ws):
        y = y - (w * y).sum(1, keepdims=True)
        X = X - (w[..., None] * X).sum(1, keepdims=True)
        xtx += np.einsum("nvk,nv,nvl->kl", X, w, X)
        xty += np.einsum("nvk,nv,nv->k", X, w, y)
        sst += (w * y * y).sum()
    b = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return 1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst, b


def main(n=16):
    P = program()
    s = P.STATS
    PIECE = WS.piece_flags()
    model = load_model("attn_s0")
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(model, W)
    lp = model_logprobs(W)
    WP = pmi_table("WPIECE", s.p_uni)
    WE = pmi_table("WEND", s.p_uni)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    cls = []
    for t in toks:
        c, _ = classes(P, t, PIECE)
        cls.append(np.array(["inside" if x.startswith("inside") else x for x in c], dtype=object))
    tables = {"pmi1": np.asarray(s.pmi[1]), "pmi2": np.asarray(s.pmi[2]), "pmi3": np.asarray(s.pmi[3]),
              "logP1": np.asarray(s.log_follow), "wpiece": WP, "wend": WE}
    SETS = [["skip_d+1"], ["pmi1"], ["pmi2"], ["pmi3"], ["logP1"], ["copy"], ["wpiece"], ["wend"],
            ["pmi1", "pmi2", "pmi3"], ["pmi1", "pmi2", "pmi3", "copy"], ["skip_d+1", "pmi1", "copy"],
            ["pmi1", "pmi2", "pmi3", "copy", "wpiece", "wend"], ["const(oracle)"], ["const(oracle)", "skip_d+1"],
            ["const(oracle)", "pmi1", "pmi2", "pmi3", "copy"]]
    for h in [0, 2, 3]:
        lpa = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        feats = []
        with torch.no_grad():
            for t in toks:
                a, _, _, _ = parts(model, torch.as_tensor(t[None]))
                a = a[0, h].double().numpy()
                T = len(t)
                ae = a * (1 - np.eye(T))
                f = {k: ae @ tab[t] for k, tab in tables.items()}
                f["skip_d+1"] = skipgram_votes(a, t, s, 0.0, 1.0, 1.0)
                cp = np.zeros((T, V))
                np.add.at(cp, (np.repeat(np.arange(T), T), np.tile(t, T)), ae.reshape(-1))
                f["copy"] = cp
                feats.append({k: v.astype(np.float32) for k, v in f.items()})
        for key in feats[0]:
            mu = np.mean([f[key].mean(0) for f in feats], axis=0)
            for f in feats:
                f[key] = f[key] - mu
        ybar = np.mean([(p - pa).mean(0) for p, pa in zip(lp, lpa)], axis=0)
        for f in feats:
            f["const(oracle)"] = np.broadcast_to(ybar.astype(np.float32), f["copy"].shape)
        print(f"head {h}", flush=True)
        for ks in SETS:
            row = []
            for c in CLASSES:
                ys_, Xs_, ws_ = [], [], []
                for p, pa, f, cc in zip(lp, lpa, feats, cls):
                    m = np.ones(len(cc), bool) if c == "all" else (cc == c)
                    ys_.append((p - pa)[m])
                    Xs_.append(np.stack([f[k][m].astype(np.float64) for k in ks], -1))
                    ws_.append(np.exp(p[m]))
                val, b = r2_set(ys_, Xs_, ws_)
                row.append(f"{c[:6]} {val:.3f}")
            print(f"   {'+'.join(ks):40s}", "  ".join(row), " b(all)", np.round(b, 2).tolist(), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
