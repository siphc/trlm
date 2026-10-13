"""ov02 exploration candidate: what do heads 0 and 2 vote for from EARLIER tokens?

Configured by environment variables (exploration only; adoptable candidates get their own files):
  OV02_ATTN    'prog' (the program's attention rules for heads 0 and 2; adoptable) or 'real' (the model's attention;
               insight only)
  OV02_TABLES  comma list of vote tables, each read as sum_{j<i} a_h(i, j) X[t_j] for h = 0 and 2:
               skip   distance-matched skip-gram PMI_{d+1} (readers.py rule; near d<8 and far d>=8 as one direction)
               big    PMI_1 of the attended token (its usual successors, as if it were the current token)
               w1,w2,w3  word-level skip-gram PMI (first token of the k-th unit after the attended token's unit)
               kpred  slot affinity: sum_a sqrt(P(prev=a|s) P(prev=a|v)) (tokens that fill the same slot as s)
               copy   one-hot of the attended token
               cs1    conditional skip-gram at d = 1 (trigram counts, K = 30): sum_b P(b|s) log(1 + n(s,b,v)/(30 P1(v|b)))
  OV02_REPLACE '1' to drop head0/head2 near+far base directions (use with 'skip' in the tables)
Tables are counted from the TRAIN stream by analysis/investigations/ov02/s03_word_tables.py (closed-form rules).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/ov02")
NAME = "ov02_votes"
HEADS = [0, 2]
ATTN = os.environ.get("OV02_ATTN", "prog")
TABLES = os.environ.get("OV02_TABLES", "big").split(",")
REPLACES = ["head0.near", "head0.far", "head2.near", "head2.far"] if os.environ.get("OV02_REPLACE") == "1" else []
OUT = "/Users/svi/Desktop/trlm/analysis/out/candidates/ov02"
_T = {}


def _table(P, name):
    if name not in _T:
        if name == "big":
            _T[name] = np.asarray(P.STATS.pmi[1], np.float64)
        elif name in ("w1", "w2", "w3"):
            _T[name] = np.load(os.path.join(OUT, f"wpmi_{name[1]}.npy")).astype(np.float64)
        elif name == "cs1":
            _T[name] = np.load(os.path.join(OUT, "cs1_K30.npy")).astype(np.float64)
        elif name == "kpred":
            _T[name] = np.load(os.path.join(OUT, "kpred.npy")).astype(np.float64)
    return _T[name]


def features(P, tokens):
    from program import head0, head2
    from program.readers import skipgram_votes
    n = len(tokens)
    if ATTN == "real":
        from common import model_attention
        A = model_attention(tokens)
        att = {0: A[0], 2: A[2]}
    else:
        att = {0: head0.attention(tokens, P.STATS), 2: head2.attention(tokens, P.STATS)}
    out = {}
    for h in (0, 2):
        a = att[h] * (1.0 - np.eye(n))
        for name in TABLES:
            if name == "skip":
                out[f"h{h}.skip"] = skipgram_votes(a, tokens, P.STATS, 0.0, 1.0, 1.0)
            elif name == "copy":
                c = np.zeros((n, 2000))
                np.add.at(c, (np.repeat(np.arange(n), n), np.tile(tokens, n)), a.reshape(-1))
                out[f"h{h}.copy"] = c
            else:
                out[f"h{h}.{name}"] = a @ _table(P, name)[tokens]
    return out
