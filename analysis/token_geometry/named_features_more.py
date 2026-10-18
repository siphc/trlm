"""Follow-ups to named_features.py (analysis only):

1. Shuffled-class control: the coverage of the shared embedding directions by the word classes, against the same
   classes with their members shuffled over tokens (class sizes kept; 20 shuffles).
2. Story-topic and valence axes (corpus only): topic axes = top eigenvectors of the symmetrised topic table
   (corpus.py: how much more often v appears 9..24 tokens after s); valence = association (PMI at distances 2..8, both
   directions) with {happy, smiled, laughed, fun, love, glad, enjoyed} minus {sad, cried, scared, angry, hurt, upset,
   bad}. Coverage of the shared directions with these added to all named features.
3. Local team vs single heads: for each seed, the union of its LOCAL heads' value codes (mean attention distance
   <= 10) against the same union in other seeds, next to single-head matches; RECALL heads likewise.

    .venv/bin/python -m analysis.token_geometry.named_features_more
Writes analysis/out/token_geometry/named_features_more.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.mech_eval import stats
from analysis.token_geometry.named_features import OUT, SEEDS, V, basis, features, model_codes


def overlap(QA, QB):
    """Share of subspace B inside subspace A (mean squared canonical correlation, over B's dimensions)."""
    return float((np.linalg.svd(QA.T @ QB, compute_uv=False) ** 2).sum() / QB.shape[1])


def topic_axes(s, k):
    T = np.asarray(s.topic, np.float64)
    T = (T + T.T) / 2
    w, U = np.linalg.eigh(T)
    return U[:, np.argsort(-np.abs(w))[:k]]


def valence(s):
    from program.vocab import ID

    pos = [" happy", " smiled", " laughed", " fun", " love", " glad", " enjoyed"]
    neg = [" sad", " cried", " scared", " angry", " hurt", " upset", " bad"]

    def assoc(words):
        ids = [ID[x] for x in words if x in ID]
        acc = np.zeros(V)
        for d in range(2, 9):
            P = np.asarray(s.pmi[d], np.float64)
            acc += P[:, ids].mean(1) + P[ids, :].T.mean(1)
        return acc / 14

    return assoc(pos) - assoc(neg)


def mean_distances(model_id):
    path = os.path.join("analysis", "out", "look_write_deep", model_id, "classes.json")
    d = json.load(open(path))
    return {c["head"]: c["mean_distance"] for c in d["heads"]}


def main():
    s = stats()
    F, fnames, fsets = features(s)
    codes = {m: model_codes(m) for m in SEEDS}
    rng = np.random.default_rng(0)
    res = {}
    cls_idx = sorted(i for k in fsets if k.startswith("classes") for i in fsets[k])
    for wname, w in {"frequency": s.p_uni / s.p_uni.sum(), "uniform": np.full(V, 1.0 / V)}.items():
        R = {}
        Qs = [basis(codes[m][0], w) for m in SEEDS]
        M = sum(Q @ Q.T for Q in Qs) / len(Qs)
        lam, U = np.linalg.eigh(M)
        lam, U = lam[::-1], U[:, ::-1]
        sh = lam > 0.8
        cov = lambda Q: float((lam[sh] * ((Q.T @ U[:, sh]) ** 2).sum(0)).sum() / lam[sh].sum())
        # 1. shuffled classes
        real = cov(basis(F[:, cls_idx], w))
        shuf = []
        for _ in range(20):
            G = F[:, cls_idx].copy()
            for system in ("ATTENTION", "CONTEXT", "TARGET"):
                cols = [fnames.index(n) for n in fnames if n.startswith(system + ":")]
                cols = [cls_idx.index(c) for c in cols]
                perm = rng.permutation(V)
                G[:, cols] = G[perm][:, cols]
            shuf.append(cov(basis(G, w)))
        R["classes: shared coverage, real vs shuffled"] = {"real": real, "shuffled mean": float(np.mean(shuf)),
                                                           "shuffled max": float(np.max(shuf))}
        allF = cov(basis(F, w))
        R["all named features"] = allF
        # 2. topic and valence axes
        for k in (8, 16, 32, 64):
            X = np.concatenate([F, topic_axes(s, k), valence(s)[:, None]], 1)
            R[f"all named + {k} topic axes + valence"] = cov(basis(X, w))
        R["topic axes (32) + valence alone"] = cov(basis(np.concatenate([topic_axes(s, 32), valence(s)[:, None]], 1), w))
        R["valence alone"] = cov(basis(valence(s)[:, None], w))
        # 3. local team vs single heads
        team, single, recall = [], [], []
        heads = {}
        for k, m in enumerate(SEEDS):
            md = mean_distances(m)
            heads[m] = {"local": [h for h in range(4) if md[h] <= 10], "recall": [h for h in range(4) if md[h] > 10]}
        for k, m in enumerate(SEEDS):
            for j, m2 in enumerate(SEEDS):
                if j == k:
                    continue
                QA = basis(np.concatenate([codes[m][1][h] for h in heads[m]["local"]], 1), w)
                QB = basis(np.concatenate([codes[m2][1][h] for h in heads[m2]["local"]], 1), w)
                team.append(overlap(QB, QA))
                for h in heads[m]["local"]:
                    Qh = basis(codes[m][1][h], w)
                    single.append(max(overlap(basis(codes[m2][1][h2], w), Qh) for h2 in heads[m2]["local"]))
                for h in heads[m]["recall"]:
                    Qh = basis(codes[m][1][h], w)
                    recall.append(max([overlap(basis(codes[m2][1][h2], w), Qh) for h2 in heads[m2]["recall"]] or [np.nan]))
        R["heads per seed"] = heads
        R["local team union: share inside other seeds' local union (mean)"] = float(np.mean(team))
        R["local single head: best other-seed local head (mean)"] = float(np.mean(single))
        tu = []
        for k, m in enumerate(SEEDS):
            for j, m2 in enumerate(SEEDS):
                if j == k:
                    continue
                QB = basis(np.concatenate([codes[m2][1][h] for h in heads[m2]["local"]], 1), w)
                for h in heads[m]["local"]:
                    tu.append(overlap(QB, basis(codes[m][1][h], w)))
        R["local single head: inside other seeds' local union (mean)"] = float(np.mean(tu))
        R["recall head: best other-seed recall head (mean)"] = float(np.nanmean(recall)) if recall else None
        # reference: a random subspace of the union's size
        d_union = int(np.mean([basis(np.concatenate([codes[m][1][h] for h in heads[m]["local"]], 1), w).shape[1] for m in SEEDS]))
        R["random reference: union dims / V"] = d_union / V
        res[wname] = R
        print(wname, json.dumps(R, indent=1), flush=True)
    with open(os.path.join(OUT, "named_features_more.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
