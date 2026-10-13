"""How one attention head is computed from its readable tables. No numbers live here.

A head at position i looks back at every position j <= i (j = i is the current token). Its preference for j is

    SALIENCE[token j]  +  PREFER[class of token i][class of token j]  +  RECENCY[bucket of i - j]

(PREFER uses the class system named by the head's PREFER_CLASSES, the ATTENTION classes) and the preferences
become attention weights through a softmax over j. Each attended token s then votes for next tokens v with
RECALL(s, v):

    CLASS_RECALL[row class of s][column class of v]  +  COPY[row class of s] if v == s  +  RECALL exceptions (s, v)

(rows and COPY use the system named by RECALL_ROWS, the CONTEXT classes; columns use RECALL_COLUMNS, the
TARGET classes). The head's contribution at position i is the attention-weighted sum of those votes. When the
head's tables say SELF_VOTES = False, a head that looks at the current token casts no vote there: everything the
current token says is in follow.py, and the head only passes on what earlier tokens say.

CLAIM (the abstention rule; all four heads have SELF_VOTES = False; tests/test_attention.py):
  Attention on the current token is an abstention. It takes part in the softmax, so it takes weight
  away from the earlier tokens, but it casts no vote: at position i a head's votes are (1 - a_ii)
  times the attention-weighted average of what the earlier tokens say.
  - At the first token of a text there is nothing earlier, so every head abstains fully and the
    prediction there is exactly FOLLOW alone.
  - On held-out text the heads abstain on 52% (head 0), 29% (head 1), 36% (head 2) and 37% (head 3)
    of their attention, at positions after the first (checked in tests/test_head0.py ... test_head3.py).
  - The rule matters: letting the heads also vote with the current token (counting it twice, once in
    FOLLOW and once in the heads) would raise the KL to the model from 0.47 to 1.89 nats, with the same
    tables (they were fitted with SELF_VOTES = False and are not refitted for this comparison).
  - Model: its heads 0 and 2 put about as much attention on the current token (49%, 34%) as program
    heads 0 and 2 abstain, but there that attention carries the current token's own vote. Model heads
    1 and 3 look at the current token less (4% and 19%).
"""
import numpy as np

from .tables_lib import TOKEN_CLASS, class_by_class, expand_classes, pairs_matrix, per_class, per_token
from .vocab import VOCAB_SIZE


class Head:
    def __init__(self, tables):
        self.salience = per_token(tables.SALIENCE)
        self.attention_class = TOKEN_CLASS[tables.PREFER_CLASSES]
        self.prefer = class_by_class(tables.PREFER, tables.PREFER_CLASSES, tables.PREFER_CLASSES)
        starts = np.asarray(tables.RECENCY_BUCKET_START)
        self.recency = np.asarray(tables.RECENCY)
        self.bucket_of_distance = np.searchsorted(starts, np.arange(256), side="right") - 1
        rows, columns = tables.RECALL_ROWS, tables.RECALL_COLUMNS
        copy = per_class(tables.COPY, rows)[TOKEN_CLASS[rows]]
        self.recall = (expand_classes(class_by_class(tables.CLASS_RECALL, rows, columns), rows, columns)
                       + np.diag(copy) + pairs_matrix(tables.RECALL))
        self.self_votes = tables.SELF_VOTES

    def weights(self, tokens):
        """Attention weights, shape (n, n); row i sums to 1 over j <= i."""
        n = len(tokens)
        cls = self.attention_class[tokens]
        dist = np.arange(n)[:, None] - np.arange(n)[None, :]
        pref = (self.salience[tokens][None, :] + self.prefer[cls][:, cls]
                + self.recency[self.bucket_of_distance[np.maximum(dist, 0)]])
        pref = np.where(dist >= 0, pref, -np.inf)
        pref = pref - pref.max(axis=1, keepdims=True)
        w = np.exp(pref)
        return w / w.sum(axis=1, keepdims=True)

    def votes(self, tokens):
        """Scores added at each position, shape (n, VOCAB_SIZE)."""
        w = self.weights(tokens)
        if not self.self_votes:
            np.fill_diagonal(w, 0.0)
        return w @ self.recall[tokens]


assert VOCAB_SIZE == 2000
