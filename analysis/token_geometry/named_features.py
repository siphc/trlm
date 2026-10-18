"""How much of the models' shared token geometry do named token properties explain? (analysis only; weights as data)

Token space: every per-token code X (2000 x d) spans a subspace of R^2000 (columns centred and weighted by w(t)).
- Seeds: attn_s0..s4. Codes: the token embedding E (d = 128) and, per head, its value code LN1(E + pbar) W_V^h
  (d = 32: which token distinctions the head reads).
- Shared geometry: eigenvectors of the mean projector M = mean_k P_k over the five seeds' embedding subspaces;
  eigenvalue ~1 = a direction in every seed's subspace, ~0.2 = in one seed's only.
- Named properties F (no model): word classes (categories.py: ATTENTION, CONTEXT, TARGET), spelling, corpus statistics
  (frequency, informativeness, continuation rate, topicality), and corpus association axes (she vs he, they vs it,
  dialogue), all from the TRAIN corpus or the token strings.
- Coverage of a direction u by F: ||P_F u||^2. Reported for the shared directions, by feature set, and per head.
- Unnamed shared directions: top eigenvectors of (I - P_F) M (I - P_F), listed with their highest- and
  lowest-loading tokens, to be named by reading.
Weights: frequency (w = p(t), what next-token KL weighs) and uniform (every token alike).

    .venv/bin/python -m analysis.token_geometry.named_features [--top 12]
Writes analysis/out/token_geometry/named_features.json and named_features_directions.txt.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import REPO
from analysis.look_write import common as LW
from analysis.mech_eval import stats

OUT = os.path.join(REPO, "analysis", "out", "token_geometry")
V = 2000
SEEDS = [f"attn_s{k}" for k in range(5)]


def basis(X, w, tol=1e-6):
    """Orthonormal basis (V x r) of the weighted, centred column space of X (V x d)."""
    X = np.asarray(X, np.float64)
    X = X - (w[:, None] * X).sum(0) / w.sum()
    Y = np.sqrt(w)[:, None] * X
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    return U[:, S > tol * S.max()]


def features(s):
    """Named token properties (V x m) and the column index of each feature set."""
    from program.categories import CLASS_SYSTEMS
    from program.vocab import ID, TOKENS

    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, np.float64)); names.append(name)

    tok = TOKENS
    strip = [t.lstrip(" ") for t in tok]
    sets = {}
    start = len(cols)
    add("space", [t.startswith(" ") for t in tok])
    add("capital", [x[:1].isupper() for x in strip])
    add("piece", s.piece)
    add("punctuation", [len(x) > 0 and not any(c.isalnum() for c in x) for x in strip])
    add("digit", [any(c.isdigit() for c in x) for x in strip])
    add("quote", ['"' in t for t in tok])
    add("newline", ["\n" in t for t in tok])
    add("end_of_story", np.arange(V) == 0)
    add("length", [min(len(x), 10) for x in strip])
    for suf in ("s", "ed", "ing", "ly", "er", "y"):
        add(f"suffix_{suf}", [x.isalpha() and len(x) > len(suf) + 1 and x.endswith(suf) for x in strip])
    sets["spelling"] = list(range(start, len(cols)))
    start = len(cols)
    add("log_frequency", np.log(s.p_uni))
    for d in range(8):
        add(f"info_{d + 1}", s.info[d])
    add("continuation_rate", s.cont)
    add("topicality", getattr(s, "topicality", np.zeros(V)))
    add("sentence_end", s.sentence_end)
    sets["corpus_scalars"] = list(range(start, len(cols)))
    start = len(cols)

    def assoc(words, forward=True):
        ids = [ID[w] for w in words if w in ID]
        acc = np.zeros(V)
        for d in range(2, 9):
            P = np.asarray(s.pmi[d], np.float64)
            acc += (P[:, ids] if forward else P[ids, :].T).mean(1)
        return acc / 7

    she, he = [" she", " her", " She", " Her"], [" he", " his", " him", " He", " His"]
    they, it = [" they", " them", " their", " They"], [" it", " its", " It"]
    quote = [' "', '"', ',"', '."', '!"', '?"']
    for name, a, b in (("she_vs_he", she, he), ("they_vs_it", they, it)):
        add(f"{name}_after", assoc(a) - assoc(b))
        add(f"{name}_before", assoc(a, False) - assoc(b, False))
    add("dialogue_after", assoc(quote))
    add("dialogue_before", assoc(quote, False))
    sets["association_axes"] = list(range(start, len(cols)))
    for system in ("ATTENTION", "CONTEXT", "TARGET"):
        start = len(cols)
        for cname, members in CLASS_SYSTEMS[system].items():
            v = np.zeros(V)
            v[[ID[m] for m in members]] = 1.0
            add(f"{system}:{cname}", v)
        sets[f"classes_{system}"] = list(range(start, len(cols)))
    return np.stack(cols, 1), names, sets


def model_codes(model_id):
    w = LW.weights(LW.load(model_id))
    x = LW.layernorm(w["wte"] + LW.pbar(w)[None], w["g1"])
    return w["wte"], [x @ w["Wv"][h] for h in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    from program.vocab import TOKENS

    os.makedirs(OUT, exist_ok=True)
    s = stats()
    F, fnames, fsets = features(s)
    codes = {m: model_codes(m) for m in SEEDS}
    weightings = {"frequency": s.p_uni / s.p_uni.sum(), "uniform": np.full(V, 1.0 / V)}
    res = {"n_features": F.shape[1], "feature_sets": {k: len(v) for k, v in fsets.items()}, "weightings": {}}
    lines = []
    nested = [("spelling", ["spelling"]), ("+corpus scalars", ["spelling", "corpus_scalars"]),
              ("+association axes", ["spelling", "corpus_scalars", "association_axes"]),
              ("+ATTENTION classes", ["spelling", "corpus_scalars", "association_axes", "classes_ATTENTION"]),
              ("+CONTEXT, TARGET classes", list(fsets))]
    for wname, w in weightings.items():
        R = {}
        Qs = [basis(codes[m][0], w) for m in SEEDS]
        M = sum(Q @ Q.T for Q in Qs) / len(Qs)
        lam, U = np.linalg.eigh(M)
        lam, U = lam[::-1], U[:, ::-1]
        R["embedding_dims_per_seed"] = [int(Q.shape[1]) for Q in Qs]
        R["mean_projector_eigenvalues"] = {"> 0.9": int((lam > 0.9).sum()), "> 0.8": int((lam > 0.8).sum()),
                                           "> 0.6": int((lam > 0.6).sum()), "> 0.4": int((lam > 0.4).sum()),
                                           "top 20": [round(float(x), 3) for x in lam[:20]]}
        shared = lam > 0.8
        # seed-to-seed: share of seed k's subspace inside seed j's (mean squared canonical correlation)
        cross = [float((np.linalg.svd(Qs[a].T @ Qs[b], compute_uv=False) ** 2).mean())
                 for a in range(5) for b in range(5) if a != b]
        R["seed_to_seed_subspace_overlap"] = {"mean": float(np.mean(cross)), "min": float(np.min(cross))}
        R["coverage"] = {}
        for label, keys in nested:
            idx = sorted(set(i for k in keys for i in fsets[k]))
            QF = basis(F[:, idx], w)
            cov_dir = ((QF.T @ U[:, shared]) ** 2).sum(0)
            per_seed = [float((np.linalg.svd(QF.T @ Q, compute_uv=False) ** 2).sum() / Q.shape[1]) for Q in Qs]
            R["coverage"][label] = {
                "n_named_dims": int(QF.shape[1]),
                "shared_directions_covered (eigenvalue-weighted)": float((lam[shared] * cov_dir).sum() / lam[shared].sum()),
                "share of each seed's embedding inside the named span (mean)": float(np.mean(per_seed)),
                "random span of the same size (expected)": float(QF.shape[1] / (V - 1))}
        classes_only = basis(F[:, sorted(i for k in fsets if k.startswith("classes") for i in fsets[k])], w)
        R["coverage"]["classes only"] = {
            "n_named_dims": int(classes_only.shape[1]),
            "shared_directions_covered (eigenvalue-weighted)": float((lam[shared] * ((classes_only.T @ U[:, shared]) ** 2).sum(0)).sum() / lam[shared].sum())}
        # heads' value codes: coverage by all named features, and by the best-matching head of each other seed
        QF = basis(F, w)
        heads = {}
        for k, m in enumerate(SEEDS):
            for h in range(4):
                Qh = basis(codes[m][1][h], w)
                named = float((np.linalg.svd(QF.T @ Qh, compute_uv=False) ** 2).sum() / Qh.shape[1])
                best = []
                for j, m2 in enumerate(SEEDS):
                    if j == k:
                        continue
                    best.append(max(float((np.linalg.svd(basis(codes[m2][1][h2], w).T @ Qh, compute_uv=False) ** 2).mean())
                                    for h2 in range(4)))
                heads[f"{m}/h{h}"] = {"named": named, "best_other_seed_head (mean over seeds)": float(np.mean(best)),
                                      "own_embedding": float((np.linalg.svd(Qs[k].T @ Qh, compute_uv=False) ** 2).sum() / Qh.shape[1])}
        R["head_value_codes"] = heads
        # unnamed shared directions
        PF = QF @ QF.T
        Mr = M - PF @ M - M @ PF + PF @ M @ PF
        lr, Ur = np.linalg.eigh(Mr)
        lr, Ur = lr[::-1], Ur[:, ::-1]
        R["unnamed_shared_eigenvalues_top"] = [round(float(x), 3) for x in lr[:args.top]]
        lines.append(f"===== weighting: {wname}; unnamed shared directions (eigenvalue = share of seeds that hold it)")
        for i in range(args.top):
            u = Ur[:, i] / np.sqrt(w)                      # loading per token (unweighted)
            u = u - (w * u).sum() / w.sum()
            score = u * np.sqrt(w)                         # weighted contribution, for ranking
            hi, lo = np.argsort(-score)[:14], np.argsort(score)[:14]
            lines.append(f"[{i}] eigenvalue {lr[i]:.3f}")
            lines.append("   + " + " ".join(repr(TOKENS[t]) for t in hi))
            lines.append("   - " + " ".join(repr(TOKENS[t]) for t in lo))
        res["weightings"][wname] = R
        print(wname, json.dumps({k: v for k, v in R.items() if k not in ("head_value_codes",)}, indent=1), flush=True)
    with open(os.path.join(OUT, "named_features.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "named_features_directions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
