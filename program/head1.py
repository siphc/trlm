"""RECALL (model head 1): earlier story words that could come next are recalled and vote for themselves, their
topic and what follows them.

Where it looks, at position i, for each j <= i (d = i - j tokens back):

    score(j) = - d / MEMORY  +  SALIENCE[group of t_j]  +  SELF_BONUS * [j = i]
               +  SUCCESSOR * class_successor(t_i, t_j)

MEMORY sets a slow recency decay (attention halves every ~23 tokens). The salience favours names and content
words over word endings, linkers and sentence ends. class_successor(q, k) (corpus.py) is high when words of k's
TARGET class often follow words of q's CONTEXT class: after '.', names and sentence openers; after ' the', nouns.

What each attended token s votes for, by its group g:

    COPY[g] * [v = s]  +  TOPIC[g] * topic(s, v)  +  NEXT[g] * PMI_1(s, v)

topic(s, v) (corpus.py) says how much more often v appears 9..24 tokens after s than usual: ' dog' -> ' owner',
' tail'; ' Lily' -> ' she', ' her'. At the first position of a window the only token it can read is the current one,
so it votes to repeat it, as the model does there.

CLAIM (tests/test_head1.py): RECALL's votes from earlier tokens change the output in the same direction as model
head 1's earlier-token part (median probability-weighted correlation >= 0.5), and more so than the same votes with
uniform attention over earlier tokens.
"""
import numpy as np

from .categories import CLASS_SYSTEMS
from .readers import softmax_rows
from .vocab import ID

NAME = "head1"
# Word groups (unions of ATTENTION classes in categories.py); every other token is a CONTENT_WORD.
GROUPS = {
    "NAMES": ["NAMES", "NAMES_UNSPACED"],
    "SENTENCE_ENDS": ["SENTENCE_ENDS"],
    "WORD_ENDS_AND_LINKERS": ["FRAGMENTS_H", "FRAGMENTS_G", "FRAGMENTS_D", "FRAGMENTS_ENDINGS", "FRAGMENTS_VC",
                              "FRAGMENTS_SHORT", "SUFFIXES_PLURAL", "SUFFIXES_ING_ED", "LETTERS_LOWER", "LINKERS",
                              "PREPOSITIONS"],
    "FUNCTION_WORDS": ["SUBJECT_PRONOUNS", "OBJECT_PRONOUNS", "LINKING_VERBS", "PARTICLES", "DETERMINERS", "MODALS",
                       "TIME_WORDS", "SENTENCE_STARTS", "ADVERBS"],
}
GROUP_NAMES = list(GROUPS) + ["CONTENT_WORDS"]
# Fitted to model head 1's own attention rows (pattern cross-entropy, 32 TRAIN windows).
MEMORY = 33.87                                                  # tokens; exp(3.5224)
SALIENCE = {"NAMES": 0.738, "SENTENCE_ENDS": -0.858, "WORD_ENDS_AND_LINKERS": -0.962, "FUNCTION_WORDS": -0.321,
            "CONTENT_WORDS": -0.047}
SELF_BONUS = 0.006
SUCCESSOR = 0.457
# Fitted to the model's log-probs with the attention above frozen.
COPY = {"NAMES": 2.614, "SENTENCE_ENDS": 1.457, "WORD_ENDS_AND_LINKERS": 1.296, "FUNCTION_WORDS": 3.307,
        "CONTENT_WORDS": 4.986}
TOPIC = {"NAMES": 3.098, "SENTENCE_ENDS": -0.756, "WORD_ENDS_AND_LINKERS": 2.340, "FUNCTION_WORDS": 1.286,
         "CONTENT_WORDS": 2.965}
NEXT = {"NAMES": -0.133, "SENTENCE_ENDS": -0.791, "WORD_ENDS_AND_LINKERS": -0.647, "FUNCTION_WORDS": -0.284,
        "CONTENT_WORDS": -0.015}


def _group_of_tokens():
    out = np.full(2000, len(GROUPS), dtype=np.int64)            # default: CONTENT_WORDS
    for g, classes in enumerate(GROUPS.values()):
        for c in classes:
            out[[ID[t] for t in CLASS_SYSTEMS["ATTENTION"][c]]] = g
    return out


GROUP = _group_of_tokens()


def _per_group(table):
    return np.array([table[g] for g in GROUP_NAMES])


def attention(tokens, stats):
    n = len(tokens)
    dist = np.arange(n)[:, None] - np.arange(n)[None, :]
    score = (-dist / MEMORY + _per_group(SALIENCE)[GROUP[tokens]][None, :] + SELF_BONUS * np.eye(n)
             + SUCCESSOR * stats.class_successor[tokens[:, None], tokens[None, :]])
    return softmax_rows(score)


def recalled_votes(tokens, stats):
    """Row j: what token t_j votes for when it is recalled."""
    g = GROUP[tokens]
    source = (_per_group(TOPIC)[g][:, None] * stats.topic[tokens] + _per_group(NEXT)[g][:, None] * stats.pmi[1][tokens])
    source[np.arange(len(tokens)), tokens] += _per_group(COPY)[g]
    return source


def votes(tokens, stats):
    return attention(tokens, stats) @ recalled_votes(tokens, stats)
