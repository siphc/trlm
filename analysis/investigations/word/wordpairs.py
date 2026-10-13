"""Word-level pair counts from the TRAIN stream (closed-form counting rules), for word starts s (non-piece tokens
followed by at least one word piece):

  W2[s, v]    v = the token two after s, when the token right after s is a piece (the word continues past s)
  WANY[s, v]  v = any word piece at offset >= 2 inside the word started by s
  WEND[s, v]  v = the first token after the word started by s (the word that s begins, once it is complete)
  WPIECE[p, v] for a word piece p at offset >= 1 inside a word: v = the token after the word ends (what follows
              the word that p is part of), counted once per (p, word)

  python -m analysis.investigations.word.wordpairs
"""
from __future__ import annotations

import os
import time

import numpy as np

from analysis.investigations.word import wordstats as WS

V = 2000


def build(corpus):
    PIECE = WS.piece_flags()
    n = len(corpus)
    W2 = np.zeros(V * V, np.int64)
    WANY = np.zeros(V * V, np.int64)
    WEND = np.zeros(V * V, np.int64)
    WPIECE = np.zeros(V * V, np.int64)
    for start in range(0, n, WS.CHUNK):
        t0 = time.time()
        hi = min(n, start + WS.CHUNK + 16)
        seg = np.asarray(corpus[start:hi]).astype(np.int64)
        m = min(WS.CHUNK, n - start)
        p = PIECE[seg]
        r = WS.run_lengths(p)
        ar = np.arange(len(seg))
        j0 = ar - r                                        # word start of each position (valid where r <= index)
        # W2
        idx = np.arange(m)
        ok = (~p[idx]) & (idx + 2 < len(seg))
        ok[ok] &= p[idx[ok] + 1]
        s_ = idx[ok]
        W2 += np.bincount(seg[s_] * V + seg[s_ + 2], minlength=V * V)
        # WANY: pieces at offset >= 2 (r >= 2), counted against the word start, for positions whose start is in-chunk
        sel = idx[(r[idx] >= 2) & (j0[idx] >= 0)]
        WANY += np.bincount(seg[j0[sel]] * V + seg[sel], minlength=V * V)
        # word ends: position e with r[e] >= 1 and next token not a piece
        e = idx[(r[idx] >= 1) & (idx + 1 < len(seg))]
        e = e[~p[e + 1]]
        e = e[j0[e] >= 0]
        WEND += np.bincount(seg[j0[e]] * V + seg[e + 1], minlength=V * V)
        # WPIECE: every piece position k in a finished word, against the token after the word
        # map each piece position to its word end: next non-piece index - 1
        nxt_np = np.minimum.accumulate(np.where(~p, ar, len(seg))[::-1])[::-1]   # next non-piece at or after k
        k = idx[(p[idx]) & (nxt_np[idx] < len(seg)) & (j0[idx] >= 0)]
        WPIECE += np.bincount(seg[k] * V + seg[nxt_np[k]], minlength=V * V)
        print(f"chunk {start // WS.CHUNK} {time.time() - t0:.1f}s", flush=True)
    return {k: v.reshape(V, V) for k, v in (("W2", W2), ("WANY", WANY), ("WEND", WEND), ("WPIECE", WPIECE))}


if __name__ == "__main__":
    from harness.common import load_corpus

    out = build(load_corpus())
    for k, v in out.items():
        np.save(os.path.join(WS.OUT, f"{k}.npy"), v)
    print({k: int(v.sum()) for k, v in out.items()})
