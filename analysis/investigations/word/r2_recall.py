"""Head-1 R^2 check for the word_recall candidate, like-for-like: one earlier-token RECALL direction (votes with the
program's own strengths, or with the strengths refitted by vote_fit), old attention vs new attention; target = model
head 1's earlier-token effect; pooled weighted R^2 overall and by position class; with and without the other parts."""
import json
import sys

import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.investigations.word.votetable import r2_set
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.vote_fit import load_candidate
from harness.common import head_means, load_model
from program import head1, phrase
from program.tests import measure as M

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
cand_path = sys.argv[2] if len(sys.argv) > 2 else "analysis/candidates/word_recall.py"
c = load_candidate(cand_path)
P = program()
s = P.STATS
PIECE = WS.piece_flags()
W = train_windows(n, seed=EVAL_SEED)
means = head_means(load_model("attn_s0"), W)
lp = model_logprobs(W)
lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
res = json.load(open(f"analysis/out/candidates/{c.NAME}.json"))
refit = res["candidate_strengths"]
base_refit = {k: v[1] for k, v in json.load(open("analysis/out/candidates/base.json"))["base_strengths"].items()}


def votes_with(strengths, prefix):
    G = head1.GROUP_NAMES
    COPY = np.array([strengths[f"{prefix}copy.{g}"] for g in G])
    TOPIC = np.array([strengths[f"{prefix}topic.{g}"] for g in G])
    NEXT = np.array([strengths[f"{prefix}next.{g}"] for g in G])

    def f(t):
        g = head1.GROUP[t]
        src = TOPIC[g][:, None] * s.topic[t] + NEXT[g][:, None] * s.pmi[1][t]
        src[np.arange(len(t)), t] += COPY[g]
        return src
    return f


prog_str = {f"head1.{k}.{g}": getattr(head1, k.upper())[g] for g in head1.GROUP_NAMES for k in ("copy", "topic", "next")}
feats, cls = [], []
for w in W:
    t = np.asarray(w[:256], dtype=np.int64)
    eye = 1 - np.eye(len(t))
    a_old = head1.attention(t, s) * eye
    a_new = c.attention(t, s) * eye
    f = {"old_prog": a_old @ head1.recalled_votes(t, s), "new_prog": a_new @ head1.recalled_votes(t, s),
         "old_refit": a_old @ votes_with(base_refit, "head1.")(t), "new_refit": a_new @ votes_with(refit, c.NAME + ".")(t),
         "local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t), "phrase": phrase.votes(t, s)}
    feats.append({k: v.astype(np.float32) for k, v in f.items()})
    cl, _ = classes(P, t, PIECE)
    cls.append(np.array(["inside" if x.startswith("inside") else x for x in cl], dtype=object))
for key in feats[0]:
    mu = np.mean([f[key].mean(0) for f in feats], axis=0)
    for f in feats:
        f[key] = f[key] - mu
for ks in (["old_prog"], ["new_prog"], ["old_refit"], ["new_refit"], ["old_refit", "local0", "local2", "phrase"],
           ["new_refit", "local0", "local2", "phrase"]):
    row = []
    for cname in ["onset", "inside", "after multi-token word", "other", "sentence end", "all"]:
        ys, Xs, ws = [], [], []
        for p, pa, f, cc in zip(lp, lpa, feats, cls):
            m = np.ones(len(cc), bool) if cname == "all" else (cc == cname)
            ys.append((p - pa)[m]); Xs.append(np.stack([f[k][m].astype(np.float64) for k in ks], -1)); ws.append(np.exp(p[m]))
        val, b = r2_set(ys, Xs, ws)
        row.append(f"{cname[:9]} {val:.3f}")
    print(f"{'+'.join(ks):36s}", "  ".join(row), flush=True)
