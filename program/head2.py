"""LOCAL-2 (model head 2): reads informative recent words, skipping word pieces and word onsets.

Where it looks, at position i, for each j <= i (d = i - j tokens back):

  the current token (j = i):
      SELF_BASE + SELF_INFO * info_1(t_i) + SELF_CONT * cont(t_i) + SELF_PIECE * piece(t_i)
  an earlier token (d >= 1):
      - DECAY * min(d - 1, 11) - DECAY_FAR * max(d - 12, 0) + PREVIOUS * [d = 1]
      + KEY_INFO * info_{min(d + 1, 8)}(t_j) + KEY_CONT * cont(t_j) + KEY_PIECE * piece(t_j)
      + KEY_SPACE * space(t_j) + KEY_SENTENCE_END * sentence_end(t_j)

with token features counted from the training stories (corpus.py): info_k(s) is how much s says about the token k
places later (KL from the unigram, nats), cont(s) the chance that the next token continues the word, piece(s) whether
s is itself a word piece. In words: it reads the current token when that token is an unfinished word or strongly
predicts what comes next; otherwise it looks back over the last few tokens (about 25% weaker per token), prefers
sentence ends and informative words, and skips word pieces and onsets, so inside a split word it reads the words
before it.

What it votes: the shared skip-gram rule of readers.py, and, from each EARLIER token s it reads,
    + NO_REPEAT * [v = s]    (expected negative: do not repeat a word just read, for example a name)

CLAIM (tests/test_head2.py): LOCAL-2's votes from earlier tokens change the output in the same direction as model
head 2's earlier-token part (median probability-weighted correlation >= 0.25), and the attention rule is what makes
them agree (uniform attention: below half of that). Removing it moves the program's output less than a quarter as
far as ablating the head moves the model's.
"""
import numpy as np

from .readers import earlier_only, mass_on_tokens, skipgram_votes, softmax_rows

NAME = "head2"
# Fitted to model head 2's own attention rows (pattern cross-entropy, 32 TRAIN windows).
SELF_BASE, SELF_INFO, SELF_CONT, SELF_PIECE = -0.336, 0.455, 0.588, -1.167
DECAY, DECAY_FAR, PREVIOUS = 0.290, 0.114, -0.081
KEY_INFO, KEY_CONT, KEY_PIECE, KEY_SPACE, KEY_SENTENCE_END = 0.529, -1.222, -1.263, -0.369, 0.846
# Fitted jointly to the model's log-probs with every attention rule frozen (analysis/vote_fit.py, v1.1).
STRENGTH_SELF, STRENGTH_NEAR, STRENGTH_FAR = 0.273, 0.608, 2.639
NO_REPEAT = -5.754


def attention(tokens, stats):
    n = len(tokens)
    dist = np.arange(n)[:, None] - np.arange(n)[None, :]
    self_score = (SELF_BASE + SELF_INFO * stats.info[0][tokens] + SELF_CONT * stats.cont[tokens]
                  + SELF_PIECE * stats.piece[tokens])
    recency = (-DECAY * np.clip(dist - 1, 0, 11) - DECAY_FAR * np.clip(dist - 12, 0, None)
               + PREVIOUS * (dist == 1))
    key = (KEY_CONT * stats.cont[tokens] + KEY_PIECE * stats.piece[tokens] + KEY_SPACE * stats.space[tokens]
           + KEY_SENTENCE_END * stats.sentence_end[tokens])
    informative = KEY_INFO * stats.info[np.clip(dist, 0, 7), tokens[None, :]]
    score = np.where(dist == 0, self_score[:, None], recency + key[None, :] + informative)
    return softmax_rows(score)


def votes(tokens, stats):
    a = attention(tokens, stats)
    return (skipgram_votes(a, tokens, stats, STRENGTH_SELF, STRENGTH_NEAR, STRENGTH_FAR)
            + NO_REPEAT * mass_on_tokens(earlier_only(a), tokens))
