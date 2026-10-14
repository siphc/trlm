"""LOCAL-0 and LOCAL-2 attention rules plus a usefulness term built from the head's OWN program vote.

    score(i, j) = program rule(i, j) + USE_NEAR * Uvote_1(t_j, t_i)  [d = 1]  + USE_FAR * Uvote_d(t_j, t_i)  [d = 2..7]
    Uvote_d = sum over the head's earlier-token vote tables X of strength_X * U^X_d      (tables.py)
    LOCAL-0: X = skip (STRENGTH_NEAR), succ (EARLIER_SUCCESSORS), alike (EARLIER_ALIKE)
    LOCAL-2: X = skip (STRENGTH_NEAR), copy (NO_REPEAT)
The vote strengths are the program's (v1.1); only USE_NEAR, USE_FAR (and, refitted with them, the rule's own
constants) are fitted, to the model head's own attention pattern.
"""
import os

import numpy as np

from program import head0, head2
from program.readers import softmax_rows

TABLES = "/Users/svi/Desktop/trlm/analysis/out/candidates/useful"
VOTE_TABLES = {"head0": {"skip": "STRENGTH_NEAR", "succ": "EARLIER_SUCCESSORS", "alike": "EARLIER_ALIKE"},
               "head2": {"skip": "STRENGTH_NEAR", "copy": "NO_REPEAT"}}
_U = {}


def vote_usefulness(name):
    """Uvote_d for d = 1..7 as a (7, V, V) float32 array, combined with the program head's vote strengths."""
    if name not in _U:
        mod = {"head0": head0, "head2": head2}[name]
        out = np.zeros((7, 2000, 2000), np.float32)
        for table, strength in VOTE_TABLES[name].items():
            for d in range(1, 8):
                out[d - 1] += np.float32(getattr(mod, strength)) * np.load(os.path.join(TABLES, f"U_{table}_{d}.npy"))
        _U[name] = out
    return _U[name]


def use_term(name, tokens, use_near, use_far):
    n = len(tokens)
    U = vote_usefulness(name)
    term = np.zeros((n, n))
    for d in range(1, min(8, n)):
        i = np.arange(d, n)
        term[i, i - d] = (use_near if d == 1 else use_far) * U[d - 1][tokens[i - d], tokens[i]]
    return term


class Head0Use:
    """LOCAL-0's rule + usefulness; constants copied from program/head0.py, plus USE_NEAR, USE_FAR."""
    __name__ = "head0_use"

    def __init__(self):
        self.SLOPE, self.SELF_BONUS = list(head0.SLOPE), list(head0.SELF_BONUS)
        self.USE_NEAR, self.USE_FAR = 0.0, 0.0

    def attention(self, tokens, stats):
        n = len(tokens)
        shape = stats.shape[tokens]
        dist = np.arange(n)[:, None] - np.arange(n)[None, :]
        score = -np.asarray(self.SLOPE)[shape][:, None] * dist + np.eye(n) * np.asarray(self.SELF_BONUS)[shape][:, None]
        return softmax_rows(score + use_term("head0", tokens, self.USE_NEAR, self.USE_FAR))


class Head2Use:
    """LOCAL-2's rule + usefulness; constants copied from program/head2.py, plus USE_NEAR, USE_FAR."""
    __name__ = "head2_use"
    CONSTS = ("SELF_BASE", "SELF_INFO", "SELF_CONT", "SELF_PIECE", "DECAY", "DECAY_FAR", "PREVIOUS", "KEY_INFO",
              "KEY_CONT", "KEY_PIECE", "KEY_SPACE", "KEY_SENTENCE_END")

    def __init__(self):
        for c in self.CONSTS:
            setattr(self, c, getattr(head2, c))
        self.USE_NEAR, self.USE_FAR = 0.0, 0.0

    def attention(self, tokens, stats):
        n = len(tokens)
        dist = np.arange(n)[:, None] - np.arange(n)[None, :]
        self_score = (self.SELF_BASE + self.SELF_INFO * stats.info[0][tokens] + self.SELF_CONT * stats.cont[tokens]
                      + self.SELF_PIECE * stats.piece[tokens])
        recency = (-self.DECAY * np.clip(dist - 1, 0, 11) - self.DECAY_FAR * np.clip(dist - 12, 0, None)
                   + self.PREVIOUS * (dist == 1))
        key = (self.KEY_CONT * stats.cont[tokens] + self.KEY_PIECE * stats.piece[tokens]
               + self.KEY_SPACE * stats.space[tokens] + self.KEY_SENTENCE_END * stats.sentence_end[tokens])
        informative = self.KEY_INFO * stats.info[np.clip(dist, 0, 7), tokens[None, :]]
        score = np.where(dist == 0, self_score[:, None], recency + key[None, :] + informative)
        return softmax_rows(score + use_term("head2", tokens, self.USE_NEAR, self.USE_FAR))


SPECS = {"head0": [("SLOPE", k, None) for k in range(4)] + [("SELF_BONUS", k, None) for k in range(4)]
                  + [("USE_NEAR", None, None), ("USE_FAR", None, None)],
         "head2": [(c, None, None) for c in Head2Use.CONSTS] + [("USE_NEAR", None, None), ("USE_FAR", None, None)]}
