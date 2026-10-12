"""Distributional clustering of the 2000 tokens from TRAIN-split bigram statistics.

Each token gets a feature vector: sqrt P(next | token) and sqrt P(prev | token) over the vocab
(Hellinger geometry), reduced by PCA. Ward clustering, then any cluster larger than MAX_SIZE is split
by k-means until every cluster fits one lint-sized table. Model-independent, so categories carry
across seeds and variants. Output: analysis/out/taxonomy_raw.json (cluster -> member token texts);
names are assigned by hand afterwards.

Usage: python -m analysis.taxonomy [n_clusters]
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "program"))
from src.train.data import load_stream  # noqa: E402

V, MAX_SIZE, PCA_DIM = 2000, 64, 96


def bigram_counts() -> np.ndarray:
    cache = os.path.join(REPO, "data", "baselines", "bigram_counts.npy")
    if os.path.exists(cache):
        return np.load(cache)
    s = load_stream("train")
    big = np.zeros(V * V, dtype=np.int64)
    step = 50_000_000
    for i in range(0, len(s) - 1, step):
        a = np.asarray(s[i : min(i + step, len(s) - 1)], dtype=np.int64)
        b = np.asarray(s[i + 1 : min(i + step, len(s) - 1) + 1], dtype=np.int64)
        big += np.bincount(a * V + b, minlength=V * V)
    big = big.reshape(V, V)
    np.save(cache, big)
    return big


def kmeans(X, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def ward(X, n):
    """Simple agglomerative Ward clustering (O(N^2) memory, fine for N=2000)."""
    clusters = {i: [i] for i in range(len(X))}
    cent = {i: X[i].copy() for i in range(len(X))}
    size = {i: 1 for i in range(len(X))}
    ids = list(clusters)
    P = np.stack([cent[i] for i in ids])
    D = ((P[:, None] - P[None]) ** 2).sum(-1) * 0.5
    np.fill_diagonal(D, np.inf)
    active = np.ones(len(ids), bool)
    sz = np.ones(len(ids))
    members = [[i] for i in ids]
    while active.sum() > n:
        a, b = np.unravel_index(np.argmin(D), D.shape)
        na, nb = sz[a], sz[b]
        P[a] = (na * P[a] + nb * P[b]) / (na + nb)
        sz[a] = na + nb
        members[a] += members[b]
        active[b] = False
        D[b, :] = np.inf
        D[:, b] = np.inf
        d = (sz[a] * sz / (sz[a] + sz)) * ((P - P[a]) ** 2).sum(-1)
        d[~active] = np.inf
        d[a] = np.inf
        D[a, :] = d
        D[:, a] = d
    return [members[i] for i in range(len(ids)) if active[i]]


def main() -> None:
    from vocab import TOKENS

    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    C = bigram_counts().astype(np.float64) + 0.01
    right = np.sqrt(C / C.sum(1, keepdims=True))
    left = np.sqrt(C.T / C.T.sum(1, keepdims=True))
    F = np.concatenate([right, left], 1)
    F -= F.mean(0)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    X = U[:, :PCA_DIM] * S[:PCA_DIM]
    groups = ward(X, n_clusters)
    out = []
    while groups:
        g = groups.pop()
        if len(g) <= MAX_SIZE:
            out.append(g)
            continue
        k = int(np.ceil(len(g) / MAX_SIZE)) + 1
        lab = kmeans(X[g], k)
        groups += [[g[i] for i in range(len(g)) if lab[i] == j] for j in range(k) if (lab == j).any()]
    freq = C.sum(1)
    out.sort(key=lambda g: -freq[g].sum())
    res = [{"size": len(g), "tokens": [TOKENS[i] for i in sorted(g, key=lambda i: -freq[i])]} for g in out]
    os.makedirs(os.path.join(REPO, "analysis", "out"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "analysis", "out", "taxonomy_raw.json"), "w"), indent=0,
              ensure_ascii=False)
    for i, r in enumerate(res):
        print(i, r["size"], " | ".join(repr(t) for t in r["tokens"][:14]))


if __name__ == "__main__":
    main()
