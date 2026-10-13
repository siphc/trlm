"""Candidate WORD RECALL: when the current token expects a word piece next (a word onset), RECALL (head 1) looks for
earlier tokens of the story that can follow it, and copies them (in-context word completion: ' bu' -> 'l' when
'bulb' was seen earlier; 'D' -> 'ave' after 'Dave').

RECALL's attention score (program/head1.py) gets one more term, for earlier tokens j < i:

    + B_WORD * cont(t_i) * max(0, PMI_1(t_i, t_j))

cont(q) = P(next token is a word piece | q) and PMI_1 the corpus bigram PMI (corpus.py): "how much the current
token expects a piece" times "how typical t_j is as the next token after t_i". B_WORD = 0.377, fitted to model head
1's OWN attention rows (analysis/investigations/word/h1_fit_word.py; pattern KL at onset queries 0.982 -> 0.853, all
queries 0.574 -> 0.564). An induction term ([t_{j-1} = t_i]) adds nothing once this term is in (0.851).

The candidate replaces RECALL's 15 vote directions by the same directions computed with the new attention.
"""
import numpy as np

NAME = "word_recall"
HEADS = [1]
B_WORD = 0.377
REPLACES = []


def _names():
    from program import head1

    return [f"head1.{kind}.{g}" for g in head1.GROUP_NAMES for kind in ("copy", "topic", "next")]


REPLACES = _names()


def attention(tokens, stats, b_word=B_WORD):
    from program import head1
    from program.readers import softmax_rows

    t = tokens
    n = len(t)
    dist = np.arange(n)[:, None] - np.arange(n)[None, :]
    score = (-dist / head1.MEMORY + head1._per_group(head1.SALIENCE)[head1.GROUP[t]][None, :]
             + head1.SELF_BONUS * np.eye(n) + head1.SUCCESSOR * stats.class_successor[t[:, None], t[None, :]])
    succ = np.maximum(np.asarray(stats.pmi[1])[t[:, None], t[None, :]], 0.0) * (dist >= 1)
    score = score + b_word * np.asarray(stats.cont)[t][:, None] * succ
    return softmax_rows(score)


def features(P, tokens):
    from program import head1

    s = P.STATS
    t = np.asarray(tokens, dtype=np.int64)
    n = len(t)
    a1 = attention(t, s)
    g = head1.GROUP[t]
    topic, nxt = np.asarray(s.topic[t], np.float64), np.asarray(s.pmi[1][t], np.float64)
    out = {}
    for k, gname in enumerate(head1.GROUP_NAMES):
        A = a1 * (g == k)[None, :]
        copy = np.zeros((n, 2000))
        np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), A.reshape(-1))
        out[f"copy.{gname}"] = copy
        out[f"topic.{gname}"] = A @ topic
        out[f"next.{gname}"] = A @ nxt
    return out
