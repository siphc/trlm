"""Statistics counted from the training stories. No numbers fitted to the model live here.

The harness hands the program the training corpus once (program.setup(corpus): the 527 million tokens of the
TinyStories training split as one stream, stories separated by the end-of-story token 0). build(corpus) counts
everything the mechanisms use, with these counting rules:

  unigram[v]           occurrences of token v;  P_uni(v) = (unigram[v] + 1) / (N + 2000)
  pairs at distance d  whole stream: (a, b) with b exactly d tokens after a, crossing story boundaries
                       within story (d >= 2): the same, but no end-of-story token strictly between a and b
  bigram rule P1(v|a)  Dirichlet-smoothed bigram, blurred with the average over a's CONTEXT class, plus a
                       unigram background (FOLLOW_SMOOTHING below; the three numbers are the best context-free
                       match to the model, fitted once)
  skip-gram PMI_d      log P_d(v | a) - log P_uni(v), P_d the within-story distance-d pairs with the same
                       Dirichlet + background smoothing (d = 2..8); PMI_1 uses the bigram rule
  topic PMI            over the first 50 million tokens: v sits 9..24 tokens after the most recent earlier s
                       (no other s in between); log((count + 100 P(v)) / (slots + 100)) - log P(v)
  class successor      max(0, log P(TARGET class of k | CONTEXT class of q) - log P(TARGET class of k)), bigram
                       counts pooled by class, add-one
  slot affinity        sum_a sqrt(P(prev = a | s) P(prev = a | v)) from the bigram counts (overlap of the two
                       tokens' predecessor distributions; 1 = they always follow the same tokens), 0 on the diagonal
  phrases              trigram and 4-gram continuation counts over the whole stream (sorted keys + counts)
  token features       info_d(s) = KL(P(token d later | s) || P(token)) from raw whole-stream pairs (d = 1..8),
                       cont(s) = P(next token is a word piece | s), and spelling flags

Everything is computed in chunks so memory stays near 2-3 GB. The program never prints (the harness talks to it
over stdout).
"""
import numpy as np

from .categories import CLASS_SYSTEMS
from .vocab import ID, TOKENS

V = 2000
CHUNK = 25_000_000
MAX_D = 8
TOPIC_TOKENS = 50_000_000        # the topic table is counted on the first 50 million tokens
TOPIC_DISTANCES = range(9, 25)   # "v sits 9..24 tokens after the most recent s"
TOPIC_PRIOR = 100.0
# FOLLOW_SMOOTHING: Dirichlet strength, class-blur weight, unigram background. Fitted once, context-free, to the
# model's average prediction after each token (exact identity; logs/mechanism_cards.md, knowledge baseline).
DIRICHLET, BLUR, BACKGROUND = 44.029, 0.036311, 0.023787


def class_index(system):
    """Class number of every token in one class system of categories.py."""
    out = np.zeros(V, dtype=np.int64)
    for k, (name, toks) in enumerate(CLASS_SYSTEMS[system].items()):
        out[[ID[t] for t in toks]] = k
    return out


def is_word_piece(text):
    return len(text) > 0 and text[0].islower() and text[0].isascii()


def shape_of(text):
    """0 word start (space + letter), 1 continuation (starts with a letter), 2 other, 3 end of story."""
    if text == "<|eos|>":
        return 3
    if text.startswith(" ") and len(text) > 1 and text[1].isalpha():
        return 0
    if text[:1].isalpha():
        return 1
    return 2


class Stats:
    """All corpus tables, as attributes (see build)."""


def build(corpus):
    s = Stats()
    n = len(corpus)
    s.n_tokens = n
    raw = np.zeros((MAX_D + 1, V * V), dtype=np.int64)       # raw[d]: whole-stream pairs at distance d
    within = np.zeros((MAX_D + 1, V * V), dtype=np.int64)    # within[d]: within-story pairs (d >= 2)
    unigram = np.zeros(V, dtype=np.int64)
    for start in range(0, n, CHUNK):
        length = min(CHUNK, n - start)
        seg = np.asarray(corpus[start : start + length + MAX_D]).astype(np.int64)
        unigram += np.bincount(seg[:length], minlength=V)
        ends = np.cumsum(seg == 0)
        for d in range(1, MAX_D + 1):
            m = min(length, len(seg) - d)
            keys = seg[:m] * V + seg[d : d + m]
            raw[d] += np.bincount(keys, minlength=V * V)
            if d >= 2:
                ok = (ends[d - 1 : d - 1 + m] - ends[:m]) == 0
                within[d] += np.bincount(keys[ok], minlength=V * V)
    s.unigram = unigram.astype(np.float64)
    s.p_uni = (s.unigram + 1.0) / (n + V)
    log_uni = np.log(s.p_uni)
    s.log_uni = log_uni.astype(np.float32)

    # FOLLOW's bigram rule and the skip-gram PMI tables
    big = raw[1].reshape(V, V).astype(np.float64)
    s.bigram_counts = big
    pdir = (big + DIRICHLET * s.p_uni) / (big.sum(1, keepdims=True) + DIRICHLET)
    ctx = class_index("CONTEXT")
    members = np.zeros((ctx.max() + 1, V))
    members[ctx, np.arange(V)] = 1.0
    class_avg = (members @ pdir) / members.sum(1, keepdims=True)
    p1 = (1 - BACKGROUND - BLUR) * pdir + BLUR * class_avg[ctx] + BACKGROUND * s.p_uni
    s.log_follow = np.log(p1).astype(np.float32)                       # log P1(v | a)
    s.pmi = {1: (np.log(p1) - log_uni).astype(np.float32)}
    for d in range(2, MAX_D + 1):
        c = within[d].reshape(V, V).astype(np.float64)
        p = (1 - BACKGROUND) * (c + DIRICHLET * s.p_uni) / (c.sum(1, keepdims=True) + DIRICHLET) + BACKGROUND * s.p_uni
        s.pmi[d] = (np.log(p) - log_uni).astype(np.float32)

    # token features: informativeness about the token d later, chance the next token is a word piece, spelling
    pu = s.unigram / n
    s.info = np.zeros((MAX_D, V))
    for d in range(1, MAX_D + 1):
        c = raw[d].reshape(V, V).astype(np.float64)
        p = c / np.maximum(c.sum(1, keepdims=True), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * (np.log(p) - np.log(np.maximum(pu, 1e-12))[None, :]), 0.0)
        s.info[d - 1] = terms.sum(1)
    del raw, within
    piece = np.array([is_word_piece(t) for t in TOKENS], dtype=np.float64)
    s.piece = piece
    s.cont = (big / np.maximum(big.sum(1, keepdims=True), 1.0)) @ piece
    s.space = np.array([t.startswith(" ") for t in TOKENS], dtype=np.float64)
    s.sentence_end = np.array([t in {".", "!", "?", '."', '!"', '?"', '".'} for t in TOKENS], dtype=np.float64)
    s.shape = np.array([shape_of(t) for t in TOKENS], dtype=np.int64)

    # class successor table: is k the kind of word that follows q?
    tgt = class_index("TARGET")
    to_tgt = np.zeros((V, tgt.max() + 1))
    to_tgt[np.arange(V), tgt] = 1.0
    cqk = big @ to_tgt                                   # (query token, TARGET class of the next token)
    ccc = members @ cqk                                  # (CONTEXT class, TARGET class)
    k_t = ccc.shape[1]
    p_k = (ccc.sum(0) + 1.0) / (ccc.sum() + k_t)
    pmi_cc = np.log((ccc + 1.0) / (ccc.sum(1, keepdims=True) + k_t)) - np.log(p_k)[None, :]
    s.class_successor = np.maximum(pmi_cc, 0.0)[ctx][:, tgt].astype(np.float32)

    # slot affinity: do s and v fill the same slot, i.e. are they preceded by the same tokens?
    root_prev = np.sqrt(big / np.maximum(big.sum(0, keepdims=True), 1.0))   # column s: sqrt(P(prev = a | s))
    s.slot_affinity = (root_prev.T @ root_prev).astype(np.float32)
    np.fill_diagonal(s.slot_affinity, 0.0)

    s.topic = topic_table(corpus)
    s.tri_keys, s.tri_counts, s.quad_keys, s.quad_counts = phrase_counts(corpus)
    return s


def topic_table(corpus):
    """PMI of v sitting 9..24 tokens after the most recent earlier occurrence of s (first 50M tokens)."""
    x = np.asarray(corpus[:TOPIC_TOKENS]).astype(np.int64)
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    gap_sorted = np.full(n, 10 ** 9, dtype=np.int64)
    same = xs[1:] == xs[:-1]
    gap_sorted[:-1] = np.where(same, order[1:] - order[:-1], 10 ** 9)
    gap = np.empty(n, dtype=np.int64)          # distance to the next occurrence of the same token
    gap[order] = gap_sorted
    del order, xs, gap_sorted, same
    counts = np.zeros(V * V, dtype=np.int64)
    slots = np.zeros(V, dtype=np.int64)
    for d in TOPIC_DISTANCES:
        keep = gap[: n - d] >= d
        src, dst = x[: n - d][keep], x[d:][keep]
        counts += np.bincount(src * V + dst, minlength=V * V)
        slots += np.bincount(src, minlength=V)
    uni = np.bincount(x, minlength=V).astype(np.float64)
    pv = (uni + 1.0) / (uni + 1.0).sum()
    c = counts.reshape(V, V).astype(np.float64)
    table = np.log((c + TOPIC_PRIOR * pv[None, :]) / (slots[:, None] + TOPIC_PRIOR)) - np.log(pv)[None, :]
    return table.astype(np.float32)


def _merge(keys, counts, new_keys, new_counts):
    if keys is None:
        return new_keys, new_counts
    allk = np.concatenate([keys, new_keys])
    uniq, inv = np.unique(allk, return_inverse=True)
    return uniq, np.bincount(inv, weights=np.concatenate([counts, new_counts])).astype(np.int64)


def phrase_counts(corpus):
    """Sorted keys and counts of every trigram (a, b, v) and 4-gram (a, b, c, v) in the stream."""
    n = len(corpus)
    k3 = c3 = k4 = c4 = None
    for start in range(0, n - 3, CHUNK):
        length = min(CHUNK, n - 3 - start)
        seg = np.asarray(corpus[start : start + length + 3]).astype(np.int64)
        a, b, c, d = seg[:length], seg[1 : length + 1], seg[2 : length + 2], seg[3 : length + 3]
        tri = (a * V + b) * V + c
        u, cnt = np.unique(tri, return_counts=True)
        k3, c3 = _merge(k3, c3, u, cnt)
        quad = tri * V + d
        u, cnt = np.unique(quad, return_counts=True)
        k4, c4 = _merge(k4, c4, u, cnt)
    last = np.asarray(corpus[n - 3 : n]).astype(np.int64)      # the final trigram is not in any chunk above
    k3, c3 = _merge(k3, c3, np.array([(last[0] * V + last[1]) * V + last[2]]), np.array([1]))
    return k3, c3, k4, c4


def continuations(keys, counts, context):
    """Next tokens and counts seen after `context` (an int key of the preceding tokens)."""
    lo = np.searchsorted(keys, context * V)
    hi = np.searchsorted(keys, context * V + V)
    return keys[lo:hi] - context * V, counts[lo:hi]
