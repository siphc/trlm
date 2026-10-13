"""Which heads do word assembly, and which corpus statistic describes what they write?

Target (as analysis/head_effects.py): the model's log-prob change when the earlier-token part of a head (or set of
heads) is replaced by its mean (harness 'earlier' ablation), centred per position under the model's probabilities.
Directions: the program parts' earlier-token votes (local0, local2, recall, phrase) and candidate word directions,
all centred over positions. Pooled probability-weighted R^2 by position class.

Candidate word directions (zero outside word-internal positions, i.e. where the current token is a piece and the
word started L-1 >= 1 tokens back):
  wp_K        log Pw_K(v | word so far) - log P1(v | t_i): Pw_K = (count(prefix, v) + K P1(v | t_i)) / (count + K)
  onset_skip  PMI_L(t_j0, v): what the stories put L tokens after the word start (the next token is L after it)
  word_skip   sum over earlier word tokens j0 <= j < i of PMI_{i+1-j}(t_j, v)

  python -m analysis.investigations.word.effects [n_windows]
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
from program.corpus import continuations
from program.tests import measure as M

V = 2000
TARGETS = {"h0": [0], "h2": [2], "h3": [3], "h1": [1], "h02": [0, 2], "h023": [0, 2, 3]}


def word_directions(P, t, PIECE, wk, wc, Ks=(3.0, 30.0, 300.0)):
    s = P.STATS
    n = len(t)
    out = {f"wp_{int(K)}": np.zeros((n, V), np.float32) for K in Ks}
    out["onset_skip"] = np.zeros((n, V), np.float32)
    out["word_skip"] = np.zeros((n, V), np.float32)
    for i in range(n):
        L, key = WS.context_key(t, i, PIECE)
        if L < 2:
            continue
        j0 = i - L + 1
        out["onset_skip"][i] = s.pmi[min(L, 8)][t[j0]]
        for j in range(j0, i):
            out["word_skip"][i] += s.pmi[min(i + 1 - j, 8)][t[j]]
        nxt, cnt = continuations(wk, wc, key)
        tot = cnt.sum()
        lp1 = s.log_follow[t[i]].astype(np.float64)
        p1 = np.exp(lp1)
        for K in Ks:
            p = K * p1 / (tot + K)
            p[nxt] += cnt / (tot + K)
            out[f"wp_{int(K)}"][i] = np.log(p) - lp1
    return out


SETS = [["program"], ["wp_3"], ["wp_30"], ["wp_300"], ["onset_skip"], ["word_skip"],
        ["program", "wp_30"], ["program", "wp_300"], ["program", "onset_skip"], ["program", "word_skip"],
        ["program", "wp_30", "onset_skip", "word_skip"]]
PROGRAM = ["local0", "local2", "recall", "phrase"]


def main(n=24):
    from harness.common import head_means, load_model
    from program import phrase

    P = program()
    PIECE = WS.piece_flags()
    wk, wc, _, _ = WS.load()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    cls = []
    feats = []
    for t in toks:
        c, Ls = classes(P, t, PIECE)
        c = np.array([x if not x.startswith("inside") else "inside" for x in c], dtype=object)
        cls.append(c)
        f = {"local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
             "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, P.STATS)}
        f = {k: v.astype(np.float32) for k, v in f.items()}
        f.update(word_directions(P, t, PIECE, wk, wc))
        feats.append(f)
    for key in feats[0]:
        mu = np.mean([f[key].mean(0) for f in feats], axis=0)
        for f in feats:
            f[key] = f[key] - mu
    names = ["inside", "onset", "after multi-token word", "other", "all"]
    allc = np.concatenate(cls)
    out = {}
    for tname, heads in TARGETS.items():
        lpa = model_logprobs(W, ablate_heads=heads, ablation="earlier", means=means)
        shift = np.concatenate([M.kl(p, pa) for p, pa in zip(lp, lpa)])
        res = {"shift": float(shift.mean()),
               "share_shift": {c: round(float(shift[allc == c].sum() / shift.sum()), 4) for c in names[:-1]},
               "r2": {}}
        for ks0 in SETS:
            ks = [k for x in ks0 for k in (PROGRAM if x == "program" else [x])]
            label = "+".join(ks0)
            res["r2"][label] = {}
            for c in names:
                xtx, xty, sst = np.zeros((len(ks), len(ks))), np.zeros(len(ks)), 0.0
                for p, pa, f, cc in zip(lp, lpa, feats, cls):
                    m = np.ones(len(cc), bool) if c == "all" else (cc == c)
                    if not m.any():
                        continue
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
        print(tname, "shift", round(res["shift"], 3), res["share_shift"], flush=True)
        for label, v in res["r2"].items():
            print(f"   {label:40s}", "  ".join(f"{c[:8]} {x['r2']:.3f}" for c, x in v.items()), flush=True)
    with open(os.path.join(WS.OUT, "effects.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
