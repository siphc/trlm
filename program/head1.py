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

RECALL always writes a vote of the same size, SIZE (its standard deviation over the vocabulary, each token weighted
by how common it is): the attention and the recalled words choose only the vote's direction. Without this, when
RECALL's attention falls on one or two words (at the start of a window, or on a single salient name) their full
vote drowns everything else, and when it is spread over many words their votes are too weak. In the model, head 1's
contribution to the output varies within a factor of about two across positions and does not grow when its attention
concentrates on one word (logs/mech_v3_report.md).

CLAIMS (tests/test_head1.py): RECALL's votes from earlier tokens change the output in the same direction as model
head 1's earlier-token part (median probability-weighted correlation >= 0.5), and more so than the same votes with
uniform attention over earlier tokens. Its vote has a fixed size: where RECALL's raw vote is largest, model head 1's
effect grows less than that raw vote, and the fixed-size vote is closer to the model's growth.
"""
import numpy as np

from .categories import CLASS_SYSTEMS
from .readers import earlier_only, softmax_rows
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
# Fitted to model head 1's own attention rows (pattern cross-entropy, 32 TRAIN windows; scripts/fit_scalars.py).
MEMORY = 33.879
SALIENCE = {"NAMES": 0.738, "SENTENCE_ENDS": -0.858, "WORD_ENDS_AND_LINKERS": -0.962,
            "FUNCTION_WORDS": -0.321, "CONTENT_WORDS": -0.047}
SELF_BONUS = 0.0048
SUCCESSOR = 0.457
# Fitted jointly to the model's log-probs with every attention rule frozen (scripts/fit_scalars.py); only their ratios
# matter, because the summed vote is rescaled to SIZE (scaled to root-mean-square 3).
COPY = {"NAMES": 3.457, "SENTENCE_ENDS": 7.553, "WORD_ENDS_AND_LINKERS": 2.653,
        "FUNCTION_WORDS": 2.305, "CONTENT_WORDS": 5.568}
TOPIC = {"NAMES": 1.610, "SENTENCE_ENDS": -0.657, "WORD_ENDS_AND_LINKERS": 3.658,
         "FUNCTION_WORDS": 0.785, "CONTENT_WORDS": 2.140}
NEXT = {"NAMES": -0.064, "SENTENCE_ENDS": -0.859, "WORD_ENDS_AND_LINKERS": -0.513,
        "FUNCTION_WORDS": -0.187, "CONTENT_WORDS": -0.131}
# Size of the summed vote; fitted with the vote strengths (scripts/fit_scalars.py).
SIZE = 0.525


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
    a = attention(tokens, stats)
    v = a @ recalled_votes(tokens, stats)
    return v * rescale(v, stats)


def earlier_votes(a, tokens, stats):
    """The part of the votes that comes from tokens before the current one, given the attention a (rescaled with
    the whole vote)."""
    rv = recalled_votes(tokens, stats)
    return (earlier_only(a) @ rv) * rescale(a @ rv, stats)


def rescale(v, stats):
    """(n, 1): SIZE / spread of each row of v (n, V); spread = standard deviation over the vocabulary, weighted by
    the unigram distribution."""
    w = stats.p_uni
    mean = v @ w
    spread = np.sqrt(((v - mean[:, None]) ** 2) @ w)
    return (SIZE / np.maximum(spread, 1e-9))[:, None]
