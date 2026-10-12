"""How one attention head is computed from its readable tables. No numbers live here.

A head at position i looks back at every position j <= i. Its preference for j is

    SALIENCE[token j]  +  PREFER[class of token i][class of token j]  +  RECENCY[bucket of i - j]

and the preferences become attention weights through a softmax over j. Each attended token s then
votes for next tokens v with RECALL(s, v):

    CLASS_RECALL[class of s][class of v]  +  COPY[class of s] if v == s  +  RECALL exceptions (s, v)

The head's contribution at position i is the attention-weighted sum of those votes.
"""
import numpy as np

from .tables_lib import TOKEN_CLASS, class_by_class, expand_classes, pairs_matrix, per_class, per_token
from .vocab import VOCAB_SIZE


class Head:
    def __init__(self, tables):
        self.salience = per_token(tables.SALIENCE)
        self.prefer = class_by_class(tables.PREFER)
        starts = np.asarray(tables.RECENCY_BUCKET_START)
        self.recency = np.asarray(tables.RECENCY)
        self.bucket_of_distance = np.searchsorted(starts, np.arange(256), side="right") - 1
        copy = per_class(tables.COPY)[TOKEN_CLASS]
        self.recall = (expand_classes(class_by_class(tables.CLASS_RECALL)) + np.diag(copy)
                       + pairs_matrix(tables.RECALL))

    def weights(self, tokens):
        """Attention weights, shape (n, n); row i sums to 1 over j <= i."""
        n = len(tokens)
        cls = TOKEN_CLASS[tokens]
        dist = np.arange(n)[:, None] - np.arange(n)[None, :]
        pref = (self.salience[tokens][None, :] + self.prefer[cls][:, cls]
                + self.recency[self.bucket_of_distance[np.maximum(dist, 0)]])
        pref = np.where(dist >= 0, pref, -np.inf)
        pref = pref - pref.max(axis=1, keepdims=True)
        w = np.exp(pref)
        return w / w.sum(axis=1, keepdims=True)

    def votes(self, tokens):
        """Scores added at each position, shape (n, VOCAB_SIZE)."""
        return self.weights(tokens) @ self.recall[tokens]


assert VOCAB_SIZE == 2000
