"""Statistics counted from the training stories. No numbers live here: everything is computed in setup().

The harness hands the program the training corpus once (program.setup(corpus): the 527 million tokens of the
TinyStories training split as one stream, stories separated by the end-of-story token 0). This module counts:

  unigram[v]          how often token v occurs
  pair_counts(d)[a, b] how often token b occurs d positions after token a (d = 1 is the ordinary bigram)
  repeats[v]          in how many stories token v occurs at least twice, out of the stories it occurs in at all

Counting rules are plain: every position of the stream counts once, pairs may cross story boundaries (the model's
training windows did too), and nothing is smoothed here; smoothing rules live with the mechanisms that use them.
"""
import numpy as np

V = 2000
CHUNK = 50_000_000


class Counts:
    def __init__(self, corpus, max_distance=1, chunk=CHUNK):
        self.n_tokens = len(corpus)
        self.unigram = np.bincount(np.asarray(corpus[: self.n_tokens]), minlength=V).astype(np.float64)
        self._pairs = {}
        for d in range(1, max_distance + 1):
            self._pairs[d] = _count_pairs(corpus, d, chunk)

    def pair_counts(self, d=1):
        """(V, V) array: entry [a, b] counts token b exactly d positions after token a."""
        return self._pairs[d]

    def conditional(self, d=1, smoothing=0.0):
        """P(token d positions later = b | token a), add-`smoothing` per cell, rows sum to 1. Rows of tokens that
        never occur in the corpus fall back to the unigram distribution."""
        c = self._pairs[d] + smoothing
        total = c.sum(axis=1, keepdims=True)
        uni = (self.unigram + 1.0) / (self.unigram + 1.0).sum()
        return np.where(total > 0, c / np.maximum(total, 1e-300), uni[None, :])


def _count_pairs(corpus, d, chunk):
    out = np.zeros(V * V, dtype=np.int64)
    n = len(corpus)
    for start in range(0, n - d, chunk):
        stop = min(start + chunk, n - d)
        a = np.asarray(corpus[start:stop]).astype(np.int64)
        b = np.asarray(corpus[start + d : stop + d]).astype(np.int64)
        out += np.bincount(a * V + b, minlength=V * V)
    return out.reshape(V, V).astype(np.float64)


def story_repeats(corpus, chunk=CHUNK):
    """For each token: number of stories containing it at least once, and at least twice."""
    once = np.zeros(V)
    twice = np.zeros(V)
    n = len(corpus)
    carry = np.zeros(0, dtype=np.int64)
    for start in range(0, n, chunk):
        block = np.concatenate([carry, np.asarray(corpus[start : min(start + chunk, n)]).astype(np.int64)])
        ends = np.flatnonzero(block == 0)
        if len(ends) == 0:
            carry = block
            continue
        story_id = np.zeros(len(block), dtype=np.int64)
        story_id[ends[:-1] + 1] = 1
        story_id = np.cumsum(story_id)
        done = ends[-1] + 1
        sid, tok = story_id[:done], block[:done]
        keys = np.unique(sid * V + tok, return_counts=True)
        toks = keys[0] % V
        np.add.at(once, toks, 1.0)
        np.add.at(twice, toks[keys[1] >= 2], 1.0)
        carry = block[done:]
    return once, twice
