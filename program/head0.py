"""LOCAL-0 (model head 0): reads the current token and the last few tokens, with a steep recency decay.

Where it looks, at position i, for each j <= i (d = i - j tokens back):

    score(j) = - SLOPE[shape of t_i] * d  +  SELF_BONUS[shape of t_i] if j = i

The shape of a token is its spelling: a word start (space + letter), a continuation piece (starts with a letter),
other (punctuation, digits, ...), or the end-of-story token. Attention odds drop by about 25% per token back after
a word start, faster inside a word, and much more slowly after punctuation; at an end-of-story token it looks almost
only at itself.

What it votes: the shared skip-gram rule of readers.py (each token read votes for what the stories put that many
tokens after it). In addition, each EARLIER token s it reads votes
    + EARLIER_SUCCESSORS * PMI_1(s, v)     for s's own usual successors (the head writes the same vector for a token
                                           whether it is the current token or an earlier one, and on the current token
                                           its job is the bigram)
    + EARLIER_ALIKE * slot_affinity(s, v)  against words that fill the same slot as s (expected negative: after
                                           reading ' cat', do not continue with ' dog')

CLAIM (tests/test_head0.py): LOCAL-0's votes from earlier tokens change the output in the same direction as model
head 0's earlier-token part (median probability-weighted correlation >= 0.18), and the attention rule is what makes
them agree (uniform attention: below half of that). It captures only a small part of the head: removing it moves the
program's output less than a quarter as far as ablating the head moves the model's.
"""
import numpy as np

from .readers import earlier_only, skipgram_votes, softmax_rows

NAME = "head0"
# Fitted to model head 0's own attention rows (pattern cross-entropy, 32 TRAIN windows); index = token shape.
SHAPES = ["word start", "continuation", "other", "end of story"]
SLOPE = [0.282, 0.363, 0.105, 0.253]          # attention score lost per token back
SELF_BONUS = [1.263, 0.504, 1.611, 3.498]     # extra score for the current token itself
# Fitted jointly to the model's log-probs with every attention rule frozen (analysis/vote_fit.py, v1.1).
STRENGTH_SELF, STRENGTH_NEAR, STRENGTH_FAR = 0.438, 0.818, 4.468
EARLIER_SUCCESSORS, EARLIER_ALIKE = 0.606, -3.914


def attention(tokens, stats):
    n = len(tokens)
    shape = stats.shape[tokens]
    dist = np.arange(n)[:, None] - np.arange(n)[None, :]
    score = -np.asarray(SLOPE)[shape][:, None] * dist + np.eye(n) * np.asarray(SELF_BONUS)[shape][:, None]
    return softmax_rows(score)


def votes(tokens, stats):
    a = attention(tokens, stats)
    earlier = earlier_only(a)
    return (skipgram_votes(a, tokens, stats, STRENGTH_SELF, STRENGTH_NEAR, STRENGTH_FAR)
            + EARLIER_SUCCESSORS * (earlier @ stats.pmi[1][tokens])
            + EARLIER_ALIKE * (earlier @ stats.slot_affinity[tokens]))
