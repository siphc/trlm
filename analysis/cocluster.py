"""CONTEXT and TARGET word classes from TRAIN bigram counts, by information-theoretic co-clustering.

FOLLOW and RECALL score a next token v after a token t through a class table T[row class of t][column class of v].
With one taxonomy for both roles, tokens must be grouped by a compromise between how they behave as contexts and
how they behave as targets. Here the two roles get separate classes:

  CONTEXT classes (table rows)     tokens after which similar things come next
  TARGET classes (table columns)   tokens that are predicted by similar contexts

found together by the co-clustering of Dhillon, Mallela & Modha (2003, "Information-theoretic co-clustering"):
minimise the loss of mutual information between the current and the next token,

  I(t; v) - I(row(t); col(v)) = sum_t f_t KL(P(. | t) || Q(. | t)),   Q(v | t) = P(col(v) | row(t)) P(v | col(v)),

which is exactly the excess KL of the best class-level bigram  WORD[v] + T[row(t), col(v)]  over the true bigram.
The algorithm alternates: every row moves to the row class whose class-level next-token distribution fits it
best, then every column likewise; each half-step can only lower the loss. Started from k-means++ on Hellinger
features; RESTARTS starts, the lowest loss is kept.

Only TRAIN-split counts are used (data/baselines/bigram_counts.npy, dev excluded); no model is involved, so the
classes carry across seeds and variants. Lint post-processing (each per-token table is written one class at a
time and each class-table row is one literal, both <= 64 numbers):
  - the 106 tokens never seen in TRAIN are pooled into their own classes in both roles;
  - classes over 64 tokens are split by frequency (most frequent 64 first);
  - TARGET: co-clustered at K_TARGET classes so that after splitting there are <= 64 classes;
  - CONTEXT: co-clustered at K_CONTEXT = 64 classes; after splitting there may be more than 64 (per-class
    tables over CONTEXT classes, such as COPY, are then written as several literals).

Usage: python -m analysis.cocluster [--scan]
  writes analysis/out/cocluster_classes.json (named classes) and regenerates program/categories.py
  (via analysis/gen_categories.py). --scan prints the loss for a range of K_TARGET and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

V, MAX = 2000, 64
K_CONTEXT = 64        # row classes of the co-clustering (before the never-seen pool and frequency splits)
K_TARGET = 52         # column classes. --scan (5 restarts each): the best loss among restarts that end with
                      # <= 64 TARGET classes after splitting is at K = 52 (0.860; K = 50: 0.865, 53: 0.879,
                      # 54: 0.892; at K = 56 every restart ends with 65-69 TARGET classes)
RESTARTS = 5
ITERS = 200
OUT = os.path.join(REPO, "analysis", "out", "cocluster_classes.json")


def bigram_counts() -> np.ndarray:
    return np.load(os.path.join(REPO, "data", "baselines", "bigram_counts.npy")).astype(np.float64)


def onehot(m: np.ndarray, k: int) -> np.ndarray:
    O = np.zeros((len(m), k))
    O[np.arange(len(m)), m] = 1.0
    return O


def compact(m: np.ndarray) -> np.ndarray:
    """Relabel to 0..k-1 (empty classes disappear)."""
    return np.unique(m, return_inverse=True)[1]


def excess_kl(J: np.ndarray, r: np.ndarray, c: np.ndarray, eps: float = 1e-300) -> float:
    """sum_t f_t KL(P(.|t) || Q(.|t)) for the class-level fit of (row map r, column map c); J sums to 1."""
    kr, kc = int(r.max()) + 1, int(c.max()) + 1
    Jrc = onehot(r, kr).T @ J @ onehot(c, kc)
    f, g = J.sum(1), J.sum(0)
    pc_r = Jrc / np.maximum(Jrc.sum(1, keepdims=True), eps)              # P(col class | row class)
    pv_c = g / np.maximum(Jrc.sum(0)[c], eps)                            # P(v | col class of v)
    logq = np.log(np.maximum(pc_r[r][:, c], eps)) + np.log(np.maximum(pv_c, eps))[None, :]
    P = J / np.maximum(f[:, None], eps)
    m = J > 0
    return float((J[m] * (np.log(P[m]) - logq[m])).sum())


def cocluster(J: np.ndarray, r: np.ndarray, c: np.ndarray, iters: int = ITERS, tol: float = 1e-7):
    """Alternating row / column reassignment (DMM 2003). Every row of J must have mass (seen tokens only)."""
    f, g = J.sum(1), J.sum(0)
    P = J / f[:, None]                    # P(v | t)
    Pc = (J / g[None, :]).T               # P(t | v)
    prev = np.inf
    for _ in range(iters):
        kr, kc = int(r.max()) + 1, int(c.max()) + 1
        Jrc = onehot(r, kr).T @ J @ onehot(c, kc)
        logp = np.log(np.maximum(Jrc / Jrc.sum(1, keepdims=True), 1e-30))       # log P(col class | row class)
        r = compact(((P @ onehot(c, kc)) @ logp.T).argmax(1))                     # best row class per row
        kr = int(r.max()) + 1
        Jrc = onehot(r, kr).T @ J @ onehot(c, kc)
        logp = np.log(np.maximum(Jrc / Jrc.sum(0, keepdims=True), 1e-30))       # log P(row class | col class)
        c = compact(((Pc @ onehot(r, kr)) @ logp).argmax(1))                      # best column class per column
        e = excess_kl(J, r, c)
        if prev - e < tol:
            break
        prev = e
    return r, c, excess_kl(J, r, c)


def kmeans_pp(X: np.ndarray, k: int, w: np.ndarray, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """Weighted k-means with k-means++ seeding; returns labels."""
    n = len(X)
    C = [X[rng.choice(n, p=w / w.sum())]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        p = w * d2
        C.append(X[rng.choice(n, p=p / p.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.stack(C)
    for _ in range(iters):
        lab = ((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = (w[m, None] * X[m]).sum(0) / w[m].sum()
    return compact(lab)


def hellinger_features(D: np.ndarray, dim: int = 96) -> np.ndarray:
    """Rows of a row-stochastic matrix as sqrt-probability vectors, reduced by PCA."""
    X = np.sqrt(D)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :dim] * S[:dim]


def split_by_frequency(m: np.ndarray, freq: np.ndarray, cap: int = MAX) -> np.ndarray:
    """Split every class with more than cap members into chunks of cap, most frequent members first."""
    m = m.copy()
    nxt = int(m.max()) + 1
    for k in range(int(m.max()) + 1):
        idx = np.nonzero(m == k)[0]
        if len(idx) > cap:
            idx = idx[np.argsort(-freq[idx], kind="stable")]
            for j in range(cap, len(idx), cap):
                m[idx[j:j + cap]] = nxt
                nxt += 1
    return m


def full_maps(rs: np.ndarray, cs: np.ndarray, seen: np.ndarray, C: np.ndarray):
    """Maps over all V tokens: seen tokens keep their co-cluster, never-seen tokens are pooled (both roles);
    then classes over 64 tokens are split by frequency (row frequency for CONTEXT, column for TARGET)."""
    def one(ms, freq):
        m = np.full(V, -1)
        m[seen] = ms
        k = int(ms.max()) + 1
        unseen = np.nonzero(~seen)[0]
        for j in range(0, len(unseen), MAX):
            m[unseen[j:j + MAX]] = k + j // MAX
        return split_by_frequency(m, freq)
    return one(rs, C.sum(1)), one(cs, C.sum(0))


def run(k_context: int, k_target: int, restarts: int = RESTARTS, verbose: bool = True):
    """Co-cluster the TRAIN bigram of the seen tokens; return the full maps and losses of the restart with the
    lowest loss among those ending with <= 64 TARGET classes (None if no restart does)."""
    C = bigram_counts()
    seen = (C.sum(1) + C.sum(0)) > 0
    Cs = C[seen][:, seen]
    J = Cs / Cs.sum()
    f, g = J.sum(1), J.sum(0)
    Xr = hellinger_features(J / f[:, None])
    Xc = hellinger_features((J / g[None, :]).T)
    best = None
    for s in range(restarts):
        rng = np.random.default_rng(s)
        r0 = kmeans_pp(Xr, k_context, f, rng)
        c0 = kmeans_pp(Xc, k_target, g, rng)
        rs, cs, e = cocluster(J, r0, c0)
        r, c = full_maps(rs, cs, seen, C)
        e_full = excess_kl(C / C.sum(), r, c)
        if verbose:
            print(f"  restart {s}: co-clusters {rs.max() + 1}x{cs.max() + 1} loss {e:.4f}; after lint "
                  f"post-processing {r.max() + 1}x{c.max() + 1} classes, loss {e_full:.4f}", flush=True)
        if c.max() + 1 > MAX:      # a class-table row would need more than one literal
            continue
        if best is None or e_full < best["loss_final"]:
            best = {"restart": s, "loss_cocluster": e, "loss_final": e_full, "r": r, "c": c}
    return best


def model_excess(r: np.ndarray, c: np.ndarray) -> float | None:
    """FOLLOW-only excess KL on the attn_s0 model's mean next-token distribution per current token (diagnostic;
    the classes themselves never see the model)."""
    md = os.path.join(REPO, "analysis", "out", "attn_s0", "mean_dists.npz")
    if not os.path.exists(md):
        return None
    d = np.load(md)
    f = d["count"] / d["count"].sum()
    P = d["big"] / np.maximum(d["big"].sum(1, keepdims=True), 1e-300)
    return excess_kl(f[:, None] * P, r, c)


# Hand names, keyed by each class's most frequent member (in its role), written after reading the members
# (they are printed by `python -m analysis.cocluster`). "_RARE" marks the less frequent part of a class that was
# split at 64 tokens. A class not listed here gets a provisional name from its two most frequent members.
CONTEXT_NAMES = {
    ".": "SENTENCE_ENDS", " the": "THE_AND_LITTLE", "\n": "LINE_STARTS_AND_SYMBOLS", " to": "TO_AND_MODALS",
    " and": "AND_AND_ADVERBS", " was": "FORMS_OF_BE", ",": "COMMAS", " She": "HE_AND_SHE",
    " Lily": "NAMES_AND_FAMILY", " teacher": "NAMES_AND_FAMILY_RARE", " her": "POSSESSIVES",
    " fun": "PHRASE_ENDINGS", " lake": "PHRASE_ENDINGS_RARE", " said": "BEFORE_A_COMMA", " a": "ARTICLE_A",
    " They": "THEY_I_WE_YOU", "er": "FRAGMENTS_1", " with": "WITH_OF_FOR_MAKE", " c": "ONSETS_1",
    " that": "CONJUNCTIONS", "ing": "VERB_ENDINGS", " some": "ADJECTIVES_AND_QUANTIFIERS",
    "s": "PLURALS_AND_OBJECT_PRONOUNS", " p": "ONSETS_2", "et": "FRAGMENTS_2", " go": "MOTION_VERBS",
    " had": "HAD_SAW_FOUND", "y": "WORD_ENDINGS", "6": "WORD_ENDINGS_RARE", " in": "PREPOSITIONS",
    " happy": "HAPPY_BACK_EXCITED", " girl": "CREATURES", "t": "FRAGMENTS_3", "l": "FRAGMENTS_4",
    " friends": "FRIENDS_AND_GROUPS", " smiled": "REACTIONS_AND_FEELINGS", " around": "AROUND_HELP_INSIDE",
    " ball": "OBJECTS", "ident": "OBJECTS_RARE", "in": "FRAGMENTS_VC", " wanted": "VERBS_BEFORE_TO",
    " it": "IT_AND_EVERYONE", "u": "ONSETS_3", " see": "BARE_VERBS_TRANSITIVE", '."': "QUOTE_ENDS",
    " like": "MENTAL_VERBS", " so": "SO_VERY_REALLY", " at": "VERBS_BEFORE_A_PERSON", "o": "FRAGMENTS_5",
    "Once": "FIXED_CONTINUATIONS", "id": "FRAGMENTS_6", " big": "SIZE_AND_COLOUR", " re": "ONSETS_4",
    " teac": "ONSETS_4_RARE", " there": "THERE_AND_IT", " too": "TOO_DO_SOMETHING", " out": "OUT_PROUD_LOTS",
    " all": "ALL_EVERY_ONE", "<|eos|>": "END_OF_STORY_AND_W", " h": "ONSETS_5", " play": "PLAY_VERBS",
    " you": "YOU_AND_PLEASE", " know": "BARE_VERBS", " \n": "BLANK_LINES_AND_BYTES", " show": "SHOW_GIVE_PICK",
    " no": "NO_ANY_THREE", "It": "IT_LET_WHAT_UNSPACED", " this": "THIS_AND_FAVORITE", "Thank": "THANK_CAN_DO",
}
TARGET_NAMES = {
    ".": "SENTENCE_ENDS", " the": "DETERMINERS_AND_OBJECT_PRONOUNS", " She": "SENTENCE_STARTS",
    " had": "PAST_VERBS", " wonder": "PAST_VERBS_RARE", " and": "CONJUNCTIONS", " to": "TO_THAT_FOR",
    " p": "ONSETS_1", " bu": "ONSETS_2", " bar": "ONSETS_3", "\n": "LINE_BREAKS_AND_QUOTE_BYTES", ",": "COMMAS",
    " Lily": "NAMES", ")": "CLOSING_BRACKET", " play": "BARE_VERBS_1", " was": "WAS_AND_IS", " a": "A_AND_AN",
    " it": "IT_YOU_SOMETHING", " in": "PREPOSITIONS", " said": "SPEECH_AND_REACTION_VERBS",
    " he": "SUBJECT_PRONOUNS", " happy": "FEELINGS_AND_QUALITIES", "ing": "SUFFIXES", "e": "FRAGMENTS_1",
    "The": "OPENERS_UNSPACED", " time": "TIME_AND_SIZE_AFTER_A", " so": "SO_VERY_NOT", "le": "FRAGMENTS_2",
    "maz": "FRAGMENTS_2_RARE", "'t": "FIXED_CONTINUATIONS", "<0xBF>": "FIXED_CONTINUATIONS_RARE",
    "t": "FRAGMENTS_3", "me": "FRAGMENTS_3_RARE", " up": "PARTICLES", "s": "S_OF_NAMED",
    " man": "ANIMALS_AND_OBJECTS", " cu": "ANIMALS_AND_OBJECTS_RARE", "ad": "FRAGMENTS_4",
    "iced": "FRAGMENTS_4_RARE", ' "': "QUOTE_THERE_BUT", "ent": "FRAGMENTS_5", "lew": "FRAGMENTS_5_RARE",
    " mom": "FAMILY_AND_BODY", '"': "NAMES_UNSPACED", " have": "HAVE_CAN_LIKE", " always": "BARE_VERBS_2",
    " all": "ADVERBS", "ar": "FRAGMENTS_6", " park": "PLACES", " girl": "CREATURES_AND_NATURE",
    " were": "WERE_ARE_WILL", " things": "MASS_AND_PLURAL_NOUNS", "u": "FRAGMENTS_7", " toy": "BELONGINGS",
    " day": "TIMES_OF_DAY", "al": "FRAGMENTS_8", " old": "PLURALS_AND_OLD", " fun": "FUN_MORE_DIFFERENT",
    " playing": "ING_FORMS", " excited": "EXCITED_MUCH_PROUD", " toys": "BODY_AND_BELONGINGS",
    " idea": "IDEA_SMILE_PLACE",
}
UNSEEN = "NEVER_IN_TRAINING_TEXT"
_SYMBOL_WORDS = {".": "PERIOD", "!": "BANG", "?": "QUESTION", ",": "COMMA", '"': "QUOTE", "'": "APOSTROPHE",
                 "\n": "NEWLINE", "-": "DASH", ":": "COLON", ";": "SEMICOLON"}


def provisional_name(tokens: list[str]) -> str:
    """A placeholder name from the two most frequent members, e.g. [' teac', ' squir'] -> 'LIKE_TEAC_SQUIR'."""
    def word(t):
        t = t.strip()
        if t in _SYMBOL_WORDS:
            return _SYMBOL_WORDS[t]
        return "".join(ch for ch in t.upper() if ch.isascii() and ch.isalnum()) or "SYMBOL"
    return "LIKE_" + "_".join(word(t) for t in tokens[:2])


def named_classes(m: np.ndarray, freq: np.ndarray, seen: np.ndarray, hand: dict) -> dict[str, list[str]]:
    """{name: member texts}: classes ordered by total frequency (most frequent first), members by frequency;
    the never-seen pools come last."""
    from program.vocab import TOKENS

    mass = np.bincount(m, weights=freq)
    groups = []
    for k in range(int(m.max()) + 1):
        idx = np.nonzero(m == k)[0]
        idx = [int(i) for i in idx[np.argsort(-freq[idx], kind="stable")]]
        groups.append((not seen[idx].any(), -mass[k], idx[0], idx))
    groups.sort(key=lambda g: g[:3])
    out, n_unseen = {}, 0
    for unseen_pool, _, _, idx in groups:
        toks = [TOKENS[i] for i in idx]
        if unseen_pool:
            n_unseen += 1
            name = f"{UNSEEN}_{n_unseen}"
        else:
            name = hand.get(toks[0]) or provisional_name(toks)
        assert name.isidentifier() and name not in out, name
        out[name] = toks
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    if args.scan:
        for kt in range(46, 57):
            b = run(K_CONTEXT, kt, verbose=False)
            if b is None:
                print(f"K_TARGET {kt}: no restart ends with <= 64 TARGET classes")
                continue
            print(f"K_TARGET {kt}: best restart {b['restart']} loss {b['loss_cocluster']:.4f} -> final "
                  f"{b['loss_final']:.4f}; classes CONTEXT {b['r'].max() + 1} TARGET {b['c'].max() + 1}", flush=True)
        return
    from analysis import gen_categories

    C = bigram_counts()
    seen = (C.sum(1) + C.sum(0)) > 0
    b = run(K_CONTEXT, K_TARGET)
    context = named_classes(b["r"], C.sum(1), seen, CONTEXT_NAMES)
    target = named_classes(b["c"], C.sum(0), seen, TARGET_NAMES)
    att = gen_categories.class_map(gen_categories.attention_classes())
    J = C / C.sum()
    report = {
        "k_context": K_CONTEXT, "k_target": K_TARGET, "restart": b["restart"],
        "n_context_classes": len(context), "n_target_classes": len(target),
        "loss_on_train_bigram": {"context_target": b["loss_final"],
                                 "attention_classes_in_both_roles": excess_kl(J, att, att)},
        "follow_only_excess_on_attn_s0_mean_bigram": {
            "context_target": model_excess(b["r"], b["c"]),
            "attention_classes_in_both_roles": model_excess(att, att)},
        "provisional_names": [n for n in list(context) + list(target) if n.startswith("LIKE_")],
    }
    print(json.dumps(report, indent=1))
    for role, classes in (("CONTEXT", context), ("TARGET", target)):
        for name, toks in classes.items():
            print(f"{role} {name} ({len(toks)}): " + " ".join(repr(t) for t in toks[:12]))
    json.dump({"report": report, "CONTEXT": context, "TARGET": target}, open(OUT, "w"), indent=0,
              ensure_ascii=False)
    gen_categories.main()


if __name__ == "__main__":
    main()
