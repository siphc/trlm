"""Word-prefix statistics of the TRAIN corpus (closed-form counting rule; no model numbers).

A token is a word piece if its text starts with a lowercase ASCII letter (program.corpus.is_word_piece). The word
so far at position i is the run t_j0 .. t_i where t_j0 is the latest non-piece token at or before i (the word
start: ' prom', 'W', '"', ...). Its length is L = i - j0 + 1 (L = 1: the current token starts a word).

Counting rule (whole TRAIN stream, chunks of 25M tokens):
  for every position i with L >= 2 (the current token is a piece inside a word that started earlier):
      count (last min(L, 4) tokens of the word so far, t_{i+1})
Keys: prefix key K = sum_k (t_{i-k} + 1) * 2001^k over k < min(L, 4) (k = 0 the current token); stored as
K * 2000 + next, sorted, with counts. continuations(keys, counts, K) from program.corpus gives the next tokens.

Also counted, for the 'after a word' context: positions i whose previous token ended a multi-token word (t_{i-1}
is the last piece of a word of length >= 2 and t_i is not a piece): count (word key of t_{i-1}'s word (last
min(L,3) tokens), t_i, t_{i+1}).  Key = (Kword * 2000 + t_i) * 2000 + next  (Kword < 2001^3 so it fits).

  python -m analysis.investigations.word.wordstats     # builds the cache under analysis/out/candidates/word/
"""
from __future__ import annotations

import os
import time

import numpy as np

from analysis.fitlib import REPO

V = 2000
B = 2001
MAXL = 4
CHUNK = 25_000_000
OUT = os.path.join(REPO, "analysis", "out", "candidates", "word")


def piece_flags():
    from program.corpus import is_word_piece
    from program.vocab import TOKENS

    return np.array([is_word_piece(t) for t in TOKENS], dtype=bool)


def run_lengths(piece_seq):
    """r[i] = number of consecutive pieces ending at i (0 if t_i is not a piece)."""
    n = len(piece_seq)
    ar = np.arange(n)
    last = np.maximum.accumulate(np.where(~piece_seq, ar, -1))
    return ar - last          # if no non-piece seen yet, this is i + 1 (treated as long)


def prefix_keys(seg, r, maxl=MAXL):
    """Key of the last min(r + 1, maxl) tokens of the word so far at every position of seg (positions whose
    lookback would leave seg get a truncated key; callers skip them)."""
    n = len(seg)
    L = np.minimum(r + 1, maxl)
    key = seg + 1
    mult = 1
    for k in range(1, maxl):
        mult *= B
        prev = np.zeros(n, dtype=np.int64)
        prev[k:] = seg[:-k] + 1
        key = key + np.where(L > k, prev * mult, 0)
    return key


def _merge(keys, counts, nk, nc):
    if keys is None:
        return nk, nc
    allk = np.concatenate([keys, nk])
    uniq, inv = np.unique(allk, return_inverse=True)
    return uniq, np.bincount(inv, weights=np.concatenate([counts, nc])).astype(np.int64)


def build(corpus):
    PIECE = piece_flags()
    n = len(corpus)
    wk = wc = ak = ac = None
    look = 8
    for start in range(0, n - 1, CHUNK):
        t0 = time.time()
        lo = max(0, start - look)
        hi = min(n, start + CHUNK + 1)
        seg = np.asarray(corpus[lo:hi]).astype(np.int64)
        p = PIECE[seg]
        r = run_lengths(p)
        key = prefix_keys(seg, r)
        off = start - lo
        m = hi - 1 - lo                      # positions with a next token
        idx = np.arange(off, m)
        # word-internal contexts
        sel = idx[r[idx] >= 1]
        u, c = np.unique(key[sel] * V + seg[sel + 1], return_counts=True)
        wk, wc = _merge(wk, wc, u, c)
        # after a multi-token word: t_{i-1} piece (r >= 1), t_i not a piece
        sel = idx[(idx >= 1)]
        sel = sel[(~p[sel]) & (r[sel - 1] >= 1)]
        kw = prefix_keys(seg, r, maxl=3)[sel - 1]
        u, c = np.unique((kw * V + seg[sel]) * V + seg[sel + 1], return_counts=True)
        ak, ac = _merge(ak, ac, u, c)
        print(f"chunk {start // CHUNK}: {len(wk)} word keys, {len(ak)} after keys, {time.time() - t0:.0f}s", flush=True)
        del seg, p, r, key
    return wk, wc, ak, ac


def load():
    """(word_keys, word_counts, after_keys, after_counts), memory-mapped."""
    names = ["word_keys", "word_counts", "after_keys", "after_counts"]
    return tuple(np.load(os.path.join(OUT, f"{k}.npy"), mmap_mode="r") for k in names)


def context_key(tokens, i, PIECE, maxl=MAXL):
    """(L, key) of the word so far at position i of `tokens` (only tokens[:i+1] used)."""
    L = 1
    while L <= i and PIECE[tokens[i - L + 1]]:
        L += 1
    # L - 1 pieces ending at i; the word start is at i - L + 1 (if inside tokens)
    if i - L + 1 < 0:
        L = i + 1
    key, mult = 0, 1
    for k in range(min(L, maxl)):
        key += (int(tokens[i - k]) + 1) * mult
        mult *= B
    return L, key


if __name__ == "__main__":
    from harness.common import load_corpus

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    wk, wc, ak, ac = build(load_corpus())
    for name, arr in (("word_keys", wk), ("word_counts", wc), ("after_keys", ak), ("after_counts", ac)):
        np.save(os.path.join(OUT, f"{name}.npy"), arr)
    print(f"done in {time.time() - t0:.0f}s; word {len(wk)} keys, after {len(ak)} keys")
