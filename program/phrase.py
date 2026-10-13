"""PHRASE (behavioural): continue the last two or three tokens as a phrase seen in the training stories.

    P3(v) = (count(t_{i-1}, t_i, v) + BACKOFF * P1(v | t_i)) / (count(t_{i-1}, t_i) + BACKOFF)
    P4(v) = (count(t_{i-2}, t_{i-1}, t_i, v) + BACKOFF * P3(v)) / (count(t_{i-2}, t_{i-1}, t_i) + BACKOFF)
    vote(v) = TRIGRAM * (log P3(v) - log P1(v | t_i)) + FOURGRAM * (log P4(v) - log P3(v))

P1 is FOLLOW's bigram rule; counts are over the whole training stream (corpus.py). The trigram is skipped at the
first position and when the current token ends a story; the 4-gram also when the previous token does.

This is a behavioural stand-in, not a single head: heads 0 and 2 make token-pair-specific choices of where to look
(mostly the previous one or two tokens) that their attention rules miss, and a phrase table reproduces their effect.

CLAIM (tests/test_phrase.py): PHRASE stands in for part of heads 0 and 2: its votes agree with the model's change
when the earlier-token parts of heads 0 and 2 are replaced by their mean (median weighted correlation >= 0.3), at
least twice as much as with head 1's (the long-range head).
"""
import numpy as np

from .corpus import V, continuations

NAME = "phrase"
BACKOFF = 300.0                    # chosen on the grid {3, 30, 300} by KL to the model
TRIGRAM, FOURGRAM = 0.284, 0.215   # fitted to the model's log-probs with all mechanisms together


def votes(tokens, stats):
    n = len(tokens)
    out = np.zeros((n, V))
    for i in range(n):
        p1 = np.exp(stats.log_follow[tokens[i]].astype(np.float64))
        p3 = p1
        if i >= 1 and tokens[i] != 0:
            p3 = _backoff(stats.tri_keys, stats.tri_counts, tokens[i - 1] * V + tokens[i], p1)
        p4 = p3
        if i >= 2 and tokens[i] != 0 and tokens[i - 1] != 0:
            p4 = _backoff(stats.quad_keys, stats.quad_counts, (tokens[i - 2] * V + tokens[i - 1]) * V + tokens[i], p3)
        out[i] = TRIGRAM * (np.log(p3) - np.log(p1)) + FOURGRAM * (np.log(p4) - np.log(p3))
    return out


def _backoff(keys, counts, context, prior):
    nxt, cnt = continuations(keys, counts, context)
    total = cnt.sum()
    if total == 0:
        return prior
    p = BACKOFF * prior / (total + BACKOFF)
    p[nxt] += cnt / (total + BACKOFF)
    return p
