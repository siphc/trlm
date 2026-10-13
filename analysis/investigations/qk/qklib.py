"""Attention-rule features for heads 0 and 2 with optional pair terms (prefix qk).

Every rule is log-linear: score(i, j) = sum_k theta_k F_k(i, j), a = softmax over j <= i. Fitting theta to a head's
own attention pattern (cross-entropy) is therefore convex.
Pair features use [k, q] tables (k = t_j, q = t_i, d = i - j) built by build_tables.py, two scalars each:
one for the previous token (d = 1) and one for d = 2..7 (zero for d = 0 and d >= 8).
"""
import os

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
TABLES = os.path.join(REPO, "analysis", "out", "candidates", "qk")
_CACHE = {}


def table(name, d):
    key = f"{name}_{d}"
    if key not in _CACHE:
        _CACHE[key] = np.load(os.path.join(TABLES, f"{key}.npy"), mmap_mode="r")
    return _CACHE[key]


def pair_feature(tokens, name, dmax=7, split=True):
    """(n, n) arrays: table_d[t_j, t_i] for d = i - j in 1..dmax."""
    n = len(tokens)
    f1, f2 = np.zeros((n, n)), np.zeros((n, n))
    for d in range(1, min(dmax, n - 1) + 1):
        if name == "IG" and d > 1:
            break
        if name != "PMI" and not os.path.exists(os.path.join(TABLES, f"{name}_{d}.npy")):
            break
        tab = table(name, d) if name != "PMI" else _CACHE["PMI"][d]
        i = np.arange(d, n)
        vals = np.asarray(tab[tokens[i - d], tokens[i]], dtype=np.float64)
        (f1 if d == 1 else f2)[i, i - d] = vals
    return (f1, f2) if split else (f1 + f2,)


def dist_matrix(n):
    return np.arange(n)[:, None] - np.arange(n)[None, :]


def base_features(head, tokens, s, keyshape=False, noinfo=False):
    """Features of the program's rule for head 0 or 2 (+ optional key-shape terms). Returns names, list of arrays,
    initial theta (the program's values)."""
    from program import head0, head2

    n = len(tokens)
    dist = dist_matrix(n)
    eye = (dist == 0).astype(np.float64)
    names, F, th = [], [], []
    if head == 0:
        shape = np.asarray(s.shape)[tokens]
        for k, nm in enumerate(head0.SHAPES):
            q = (shape == k).astype(np.float64)[:, None]
            names += [f"slope[{nm}]", f"self[{nm}]"]
            F += [-dist * q, eye * q]
            th += [head0.SLOPE[k], head0.SELF_BONUS[k]]
    elif head == 3:      # generic short-range rule: free distance profile d = 0..11, linear tail, shapes
        shape = np.asarray(s.shape)[tokens]
        for d in range(12):
            names.append(f"dist[{d}]")
            F.append((dist == d) * 1.0)
            th.append(0.0)
        names.append("tail")
        F.append(-np.clip(dist - 11, 0, None) * 1.0)
        th.append(0.3)
        for k, nm in enumerate(["word start", "continuation", "other"]):
            q = (shape == k).astype(np.float64)
            names += [f"self[{nm}]", f"key[{nm}]"]
            F += [eye * q[:, None], (dist >= 1) * q[None, :]]
            th += [0.0, 0.0]
    else:
        info0 = np.asarray(s.info)[0][tokens]
        cont, piece = np.asarray(s.cont)[tokens], np.asarray(s.piece)[tokens]
        space, send = np.asarray(s.space)[tokens], np.asarray(s.sentence_end)[tokens]
        earlier = (dist >= 1).astype(np.float64)
        infok = np.asarray(s.info)[np.clip(dist, 0, 7), tokens[None, :]]
        names += ["SELF_BASE", "SELF_INFO", "SELF_CONT", "SELF_PIECE", "DECAY", "DECAY_FAR", "PREVIOUS",
                  "KEY_INFO", "KEY_CONT", "KEY_PIECE", "KEY_SPACE", "KEY_SENTENCE_END"]
        F += [eye, eye * info0[:, None], eye * cont[:, None], eye * piece[:, None],
              -np.clip(dist - 1, 0, 11) * earlier, -np.clip(dist - 12, 0, None) * earlier, (dist == 1) * 1.0,
              infok * earlier, earlier * cont[None, :], earlier * piece[None, :], earlier * space[None, :],
              earlier * send[None, :]]
        th += [head2.SELF_BASE, head2.SELF_INFO, head2.SELF_CONT, head2.SELF_PIECE, head2.DECAY, head2.DECAY_FAR,
               head2.PREVIOUS, head2.KEY_INFO, head2.KEY_CONT, head2.KEY_PIECE, head2.KEY_SPACE, head2.KEY_SENTENCE_END]
    if noinfo and "KEY_INFO" in names:
        k = names.index("KEY_INFO")
        names, F, th = names[:k] + names[k + 1:], F[:k] + F[k + 1:], th[:k] + th[k + 1:]
    if keyshape and head == 0:
        shape = np.asarray(s.shape)[tokens]
        earlier = (dist >= 1).astype(np.float64)
        for k, nm in enumerate(["word start", "continuation", "other"]):      # 'end of story' is the reference
            names.append(f"key[{nm}]")
            F.append(earlier * (shape == k).astype(np.float64)[None, :])
            th.append(0.0)
    return names, F, th


def pair_features(tokens, s, pairs):
    names, F = [], []
    if "PMI" in pairs and "PMI" not in _CACHE:
        _CACHE["PMI"] = s.pmi
    for p in pairs:
        if p.endswith("*"):                     # one scalar for all d = 1..7
            names += [f"{p[:-1]}[d=1..7]"]
            F += [sum(pair_feature(tokens, p[:-1]))]
            continue
        f1, f2 = pair_feature(tokens, p)
        names += [f"{p}[d=1]"]
        F += [f1]
        if p != "IG":
            names += [f"{p}[d=2..7]"]
            F += [f2]
    return names, F


def softmax_rows(score):
    n = score.shape[0]
    score = np.where(np.tril(np.ones((n, n), dtype=bool)), score, -np.inf)
    score = score - score.max(1, keepdims=True)
    w = np.exp(score)
    return w / w.sum(1, keepdims=True)


def attention(head, tokens, s, theta_named, keyshape=False, pairs=(), noinfo=False):
    names, F, _ = base_features(head, tokens, s, keyshape, noinfo)
    pn, pF = pair_features(tokens, s, pairs)
    names, F = names + pn, F + pF
    score = sum(theta_named[nm] * f for nm, f in zip(names, F))
    return softmax_rows(score)
