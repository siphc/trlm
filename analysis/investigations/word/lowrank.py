"""Hypothesis: a head can only write a rank-32 vote (32-dim value space, tied 128-dim embeddings), so what it writes
is a low-rank summary of the corpus table, not the full table. Test with each head's REAL attention (insight only):
direction sum_{j<i} a_ij T_k(t_j, .) where T_k is the rank-k truncated SVD of the corpus table (PMI_1 / PMI_2 /
skip PMI_{d+1}), plain or frequency-weighted (rows and columns scaled by sqrt(P(token)) before the SVD).

  python -m analysis.investigations.word.lowrank [n_windows]
"""
import sys

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.decomp import parts
from analysis.investigations.word.diag import classes
from analysis.investigations.word.votetable import r2_set
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from harness.common import head_means, load_model

V = 2000


def lowrank(T, k, wr=None, wc=None):
    T = np.asarray(T, np.float64)
    wr = np.ones(V) if wr is None else wr
    wc = np.ones(V) if wc is None else wc
    M = (T - T.mean(1, keepdims=True)) * np.sqrt(wr)[:, None] * np.sqrt(wc)[None, :]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = (U[:, :k] * S[:k]) @ Vt[:k]
    return (R / np.sqrt(wr)[:, None] / np.sqrt(wc)[None, :]).astype(np.float32)


def main(n=16):
    P = program()
    s = P.STATS
    PIECE = WS.piece_flags()
    model = load_model("attn_s0")
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(model, W)
    lp = model_logprobs(W)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    cls = []
    for t in toks:
        c, _ = classes(P, t, PIECE)
        cls.append(np.array(["inside" if x.startswith("inside") else x for x in c], dtype=object))
    pu = np.asarray(s.p_uni)
    tabs = {}
    for name, T in (("pmi1", s.pmi[1]), ("pmi2", s.pmi[2]), ("pmi3", s.pmi[3])):
        tabs[f"{name}_full"] = np.asarray(T, np.float32)
        for k in (8, 32, 128):
            tabs[f"{name}_r{k}"] = lowrank(T, k)
            tabs[f"{name}_w{k}"] = lowrank(T, k, pu, pu)
        print("svd", name, flush=True)
    for h in [0, 2, 3]:
        lpa = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        A = []
        with torch.no_grad():
            for t in toks:
                a, _, _, _ = parts(model, torch.as_tensor(t[None]))
                A.append(a[0, h].double().numpy() * (1 - np.eye(len(t))))
        print(f"head {h}", flush=True)
        for name in list(tabs) + ["pmi1_w32+pmi2_w32+pmi3_w32", "pmi1_full+pmi2_full+pmi3_full"]:
            ks = name.split("+")
            feats = [{k: (a @ tabs[k][t]) for k in ks} for a, t in zip(A, toks)]
            for k in ks:
                mu = np.mean([f[k].mean(0) for f in feats], axis=0)
                for f in feats:
                    f[k] = f[k] - mu
            row = []
            for c in ["inside", "other", "all"]:
                ys_, Xs_, ws_ = [], [], []
                for p, pa, f, cc in zip(lp, lpa, feats, cls):
                    m = np.ones(len(cc), bool) if c == "all" else (cc == c)
                    ys_.append((p - pa)[m])
                    Xs_.append(np.stack([f[k][m] for k in ks], -1))
                    ws_.append(np.exp(p[m]))
                val, b = r2_set(ys_, Xs_, ws_)
                row.append(f"{c} {val:.3f}")
            print(f"   {name:32s}", "  ".join(row), " b", np.round(b, 2).tolist(), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
