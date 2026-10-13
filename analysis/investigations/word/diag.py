"""Diagnostics: how close is the model, inside split words, to the corpus distribution given the word so far?

For 64 TRAIN windows (seed 2000): position classes by word structure, the program's residual KL by class, and at
word-internal positions KL(model || corpus word-prefix distribution) for several smoothing strengths.

  python -m analysis.investigations.word.diag
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.corpus import continuations
from program.tests import measure as M

V = 2000


def classes(P, t, PIECE):
    s = P.STATS
    n = len(t)
    out = np.empty(n, dtype=object)
    Ls = np.zeros(n, dtype=int)
    for i in range(n):
        L, _ = WS.context_key(t, i, PIECE)
        Ls[i] = L
        if i == 0:
            out[i] = "pos0"
        elif L >= 2:
            out[i] = f"inside L={min(L, 4)}"
        elif s.cont[t[i]] > 0.5:
            out[i] = "onset"
        elif PIECE[t[i - 1]] and i >= 2 and Ls[i - 1] >= 2:
            out[i] = "after multi-token word"
        elif s.sentence_end[t[i]] > 0:
            out[i] = "sentence end"
        else:
            out[i] = "other"
    return out, Ls


def word_dist(P, t, i, K, PIECE, keys, counts):
    p1 = np.exp(P.STATS.log_follow[t[i]].astype(np.float64))
    L, key = WS.context_key(t, i, PIECE)
    if L < 2:
        return p1, 0
    nxt, cnt = continuations(keys, counts, key)
    tot = cnt.sum()
    if tot == 0:
        return p1, 0
    p = K * p1 / (tot + K)
    p[nxt] += cnt / (tot + K)
    return p, tot


def main(n=64):
    P = program()
    PIECE = WS.piece_flags()
    wk, wc, _, _ = WS.load()
    W = train_windows(n, seed=EVAL_SEED)
    lp = model_logprobs(W)
    Ks = [0.3, 3.0, 30.0, 300.0]
    rows = {}
    ent = {}
    for w, p in zip(W, lp):
        t = np.asarray(w[:256], dtype=np.int64)
        q = P.predict(t)
        kl_prog = M.kl(p, q)
        cl, Ls = classes(P, t, PIECE)
        p1l = P.STATS.log_follow[t].astype(np.float64)
        kl_big = M.kl(p, p1l - np.log(np.exp(p1l).sum(1, keepdims=True)))
        for i in range(len(t)):
            c = cl[i]
            r = rows.setdefault(c, {"n": 0, "kl_prog": 0.0, "kl_bigram": 0.0, **{f"kl_word_K{K}": 0.0 for K in Ks},
                                    "seen": 0, "model_entropy": 0.0, "word_entropy_K3": 0.0})
            r["n"] += 1
            r["kl_prog"] += kl_prog[i]
            r["kl_bigram"] += kl_big[i]
            pm = np.exp(p[i])
            r["model_entropy"] += -(pm * p[i]).sum()
            if c.startswith("inside"):
                for K in Ks:
                    pw, tot = word_dist(P, t, i, K, PIECE, wk, wc)
                    r[f"kl_word_K{K}"] += float((pm * (p[i] - np.log(pw))).sum())
                    if K == 3.0:
                        r["seen"] += tot > 0
                        r["word_entropy_K3"] += -(pw * np.log(pw)).sum()
    total = sum(r["kl_prog"] for r in rows.values())
    out = {}
    for c, r in sorted(rows.items()):
        m = r["n"]
        out[c] = {"positions": m, "share_positions": round(m / sum(x["n"] for x in rows.values()), 4),
                  "share_residual": round(r["kl_prog"] / total, 4),
                  **{k: round(v / m, 4) for k, v in r.items() if k not in ("n",)}}
        print(c, out[c], flush=True)
    os.makedirs(WS.OUT, exist_ok=True)
    with open(os.path.join(WS.OUT, "diag.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
