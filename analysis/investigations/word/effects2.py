"""Like effects.py, with what the OV probe suggests: the word start read from inside the word votes in three ways,
  ws_pmi1  PMI_1(t_j0, v)   the word start's own continuations ("the word goes on")        [head 2's OV]
  ws_skip  PMI_L(t_j0, v)   what sits L tokens after the word start (the next token)       [head 0's OV]
  ws_wend  WEND-PMI(t_j0, v) what follows a finished word that starts with t_j0           [heads 3, 0]
and the earlier pieces j0 < j < i vote
  pc_pmi1  PMI_1(t_j, v),   pc_wpiece WPIECE-PMI(t_j, v) (what follows the word the piece belongs to)
All zero outside word-internal positions (L >= 2).

  python -m analysis.investigations.word.effects2 [n_windows]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M

V = 2000
TARGETS = {"h0": [0], "h2": [2], "h3": [3], "h02": [0, 2], "h023": [0, 2, 3]}
PROGRAM = ["local0", "local2", "recall", "phrase"]
NEW = ["ws_pmi1", "ws_skip", "ws_wend", "pc_pmi1", "pc_wpiece"]
SETS = [["program"]] + [[k] for k in NEW] + [["program", k] for k in NEW] + [["program"] + NEW, NEW,
                                                                            ["program", "ws_pmi1", "ws_skip", "ws_wend"]]


def pmi_table(name, p, alpha=44.0):
    C = np.load(os.path.join(WS.OUT, f"{name}.npy")).astype(np.float64)
    tot = C.sum(1, keepdims=True)
    return (np.log((C + alpha * p[None, :]) / (tot + alpha)) - np.log(p)[None, :]).astype(np.float32)


def directions(P, t, PIECE, T):
    s = P.STATS
    n = len(t)
    out = {k: np.zeros((n, V), np.float32) for k in NEW}
    for i in range(n):
        L, _ = WS.context_key(t, i, PIECE)
        if L < 2:
            continue
        j0 = i - L + 1
        out["ws_pmi1"][i] = s.pmi[1][t[j0]]
        out["ws_skip"][i] = s.pmi[min(L, 8)][t[j0]]
        out["ws_wend"][i] = T["WEND"][t[j0]]
        for j in range(j0 + 1, i):
            out["pc_pmi1"][i] += s.pmi[1][t[j]]
            out["pc_wpiece"][i] += T["WPIECE"][t[j]]
    return out


def main(n=24):
    from harness.common import head_means, load_model
    from program import phrase

    P = program()
    PIECE = WS.piece_flags()
    T = {k: pmi_table(k, P.STATS.p_uni) for k in ["WEND", "WPIECE"]}
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    cls, feats = [], []
    for t in toks:
        c, Ls = classes(P, t, PIECE)
        cls.append(np.array(["inside" if x.startswith("inside") else x for x in c], dtype=object))
        f = {"local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
             "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, P.STATS)}
        f = {k: v.astype(np.float32) for k, v in f.items()}
        f.update(directions(P, t, PIECE, T))
        feats.append(f)
    for key in feats[0]:
        mu = np.mean([f[key].mean(0) for f in feats], axis=0)
        for f in feats:
            f[key] = f[key] - mu
    names = ["inside", "all"]
    out = {}
    for tname, heads in TARGETS.items():
        lpa = model_logprobs(W, ablate_heads=heads, ablation="earlier", means=means)
        res = {"r2": {}}
        for ks0 in SETS:
            ks = [k for x in ks0 for k in (PROGRAM if x == "program" else [x])]
            label = "+".join(ks0)
            res["r2"][label] = {}
            for c in names:
                xtx, xty, sst = np.zeros((len(ks), len(ks))), np.zeros(len(ks)), 0.0
                for p, pa, f, cc in zip(lp, lpa, feats, cls):
                    m = np.ones(len(cc), bool) if c == "all" else (cc == c)
                    w = np.exp(p[m])
                    y = (p - pa)[m]
                    y = y - (w * y).sum(1, keepdims=True)
                    X = np.stack([f[k][m].astype(np.float64) for k in ks], -1)
                    X = X - (w[..., None] * X).sum(1, keepdims=True)
                    xtx += np.einsum("nvk,nv,nvl->kl", X, w, X)
                    xty += np.einsum("nvk,nv,nv->k", X, w, y)
                    sst += (w * y * y).sum()
                beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
                r2 = 1 - (sst - 2 * beta @ xty + beta @ xtx @ beta) / sst
                res["r2"][label][c] = {"r2": round(float(r2), 4), "beta": [round(float(b), 3) for b in beta]}
        out[tname] = res
        print(tname, flush=True)
        for label, v in res["r2"].items():
            print(f"   {label:58s}", "  ".join(f"{c} {x['r2']:.3f}" for c, x in v.items()),
                  " beta(inside)", v["inside"]["beta"][-5:], flush=True)
    with open(os.path.join(WS.OUT, "effects2.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
