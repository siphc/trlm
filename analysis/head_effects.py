"""How much of each model head's effect from earlier tokens does the program explain, and where?

Target: d_model = lp - lp_ablated, the change of the model's log-probs when a head's earlier-token part is
replaced by its mean (harness 'earlier' ablation), centred per position under the model's probabilities.
Explanatory directions: the earlier-token votes of each program part (centred over positions), LOCAL-0, LOCAL-2,
RECALL, and PHRASE's votes. Reported: pooled weighted R^2 (weights = model probabilities) for several sets of
directions, overall and by position class, and each class's share of the head's KL shift.

  python -m analysis.head_effects
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.fitlib import REPO, train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M

TARGETS = {"h0": [0], "h1": [1], "h2": [2], "h3": [3], "h02": [0, 2]}
SETS = [["local0"], ["local2"], ["recall"], ["phrase"], ["local0", "local2"], ["local0", "local2", "phrase"],
        ["local0", "local2", "recall", "phrase"]]


def position_classes(P, t):
    s = P.STATS
    piece = s.piece[t] > 0
    onset = (s.cont[t] > 0.5) & ~piece
    sent = s.sentence_end[t] > 0
    out = np.full(len(t), "other", dtype=object)
    out[sent] = "after sentence end"
    out[onset] = "word onset (next is likely a piece)"
    out[piece] = "current token is a word piece"
    out[: 1] = "window position 0"
    return out


def main(n=32):
    from harness.common import head_means, load_model
    from program import phrase

    P = program()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    feats = [{"local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
              "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, P.STATS)} for t in toks]
    for key in feats[0]:
        mu = np.mean([f[key].mean(0) for f in feats], axis=0)
        for f in feats:
            f[key] = f[key] - mu
    cls = [position_classes(P, t) for t in toks]
    names = sorted(set(np.concatenate(cls)))
    out = {}
    for tname, heads in TARGETS.items():
        lpa = model_logprobs(W, ablate_heads=heads, ablation="earlier", means=means)
        res = {"shift": {}, "r2": {}}
        allc = np.concatenate(cls)
        shift = np.concatenate([M.kl(p, pa) for p, pa in zip(lp, lpa)])
        res["shift"]["all"] = float(shift.mean())
        for c in names:
            res["shift"][c] = {"share_positions": float((allc == c).mean()),
                               "share_shift": float(shift[allc == c].sum() / shift.sum())}
        for ks in SETS:
            label = "+".join(ks)
            res["r2"][label] = {}
            for c in ["all"] + names:
                xtx, xty, sst = np.zeros((len(ks), len(ks))), np.zeros(len(ks)), 0.0
                for p, pa, f, cc in zip(lp, lpa, feats, cls):
                    m = slice(None) if c == "all" else (cc == c)
                    w = np.exp(p[m])
                    y = (p - pa)[m]
                    y = y - (w * y).sum(1, keepdims=True)
                    X = np.stack([f[k][m] for k in ks], -1)
                    X = X - (w[..., None] * X).sum(1, keepdims=True)
                    xtx += np.einsum("nvk,nv,nvl->kl", X, w, X)
                    xty += np.einsum("nvk,nv,nv->k", X, w, y)
                    sst += (w * y * y).sum()
                if sst == 0:
                    continue
                beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
                r2 = 1 - (sst - 2 * beta @ xty + beta @ xtx @ beta) / sst
                res["r2"][label][c] = {"r2": round(float(r2), 4), "beta": [round(float(b), 3) for b in beta]}
        out[tname] = res
        print(tname, json.dumps(res["shift"]), flush=True)
        for label, v in res["r2"].items():
            print("   ", label, {c: x["r2"] for c, x in v.items()}, flush=True)
    with open(os.path.join(REPO, "analysis", "out", "head_effects.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
