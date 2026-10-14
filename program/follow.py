"""FOLLOW: what usually comes right after the current token, from the training stories.

    vote(v) = STRENGTH * log P1(v | current token) + UNIGRAM * log P(v)

P1 is the corpus bigram with light smoothing (corpus.py): mostly the token's own continuations, a little of the
average over words of the same CONTEXT class, and a little unigram background.

In the model there is no separate bigram part: the current token's own vote travels through the direct path and
through heads 0 and 2 attending to the current token, and only their sum is a clean corpus statistic. FOLLOW is
that sum.

CLAIM (tests/test_follow.py): FOLLOW is what the model predicts from the current token alone. On its own it is much
closer to the model with the earlier-token parts of all four heads replaced by their mean (KL <= 1.35) than to the
full model (at most 0.8 times that KL).
"""
NAME = "follow"
STRENGTH = 0.390   # weight of log P1; fitted jointly with all mechanisms (scripts/fit_scalars.py)
UNIGRAM = 0.486    # weight of log P(v); same fit


def votes(tokens, stats):
    return STRENGTH * stats.log_follow[tokens] + UNIGRAM * stats.log_uni[None, :]
