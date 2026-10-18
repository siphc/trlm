"""FOLLOW-variant candidate (mechanism program round 3).

Similarity-based smoothing of FOLLOW (Dagan, Lee & Pereira 1999), counted from the corpus.

  input side:  a token borrows the successors of tokens that are followed by the same things
      B(s, r) = sum_v sqrt(P1(v | s) P1(v | r))          (Bhattacharyya overlap of successor distributions), r != s
      P_in(v | s) = sum_r B(s, r)^K P1(v | r) / sum_r B(s, r)^K
  output side: a token is expected where tokens that fill the same slot are expected
      A(u, v) = slot_affinity (corpus.py: overlap of predecessor distributions), u != v
      P_out(v | s) = sum_u P1(u | s) A(u, v)^K / sum_v (...)
Directions log P_in, log P_out for K = 2 and 8 (the fit chooses); everything else kept.
"""
import numpy as np

NAME = "sim_smooth"
HEADS = []
REPLACES = []
_T = {}


def tables(s):
    if not _T:
        p1 = np.exp(np.asarray(s.log_follow, np.float64))
        r = np.sqrt(p1)
        B = r @ r.T
        np.fill_diagonal(B, 0.0)
        A = np.asarray(s.slot_affinity, np.float64).copy()
        np.fill_diagonal(A, 0.0)
        for K in (2, 8):
            w = B ** K
            w /= w.sum(1, keepdims=True)
            _T[f"in{K}"] = np.log(w @ p1 + 1e-6).astype(np.float32)
            a = A ** K
            po = p1 @ a
            po /= po.sum(1, keepdims=True)
            _T[f"out{K}"] = np.log(po + 1e-6).astype(np.float32)
    return _T


def features(P, tokens):
    T = tables(P.STATS)
    return {f"{k}.self": np.asarray(v[tokens], np.float64) for k, v in T.items()}
