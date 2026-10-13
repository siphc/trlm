"""Head 3's attention rule from its circuit card (logs/mechanism_cards.md; 27 scalars fitted by pattern KL to head 3's
OWN attention pattern, analysis/scratch/head3_card/fit_hand.py). Pure numpy; depends on tokens[:i+1] only."""
import numpy as np

GROUPS = {
    "BOUNDARY": ["SENTENCE_ENDS", "SENTENCE_STARTS", "OPENERS_UNSPACED", "QUOTE_OPENERS", "INTERJECTIONS"],
    "LINKER": ["LINKERS"],
    "VERB": ["VERBS_BEFORE_TO", "PAST_VERBS_TRANSITIVE", "PAST_VERBS_MOTION", "SPEECH_VERBS", "NEGATED_AUXILIARIES",
             "MODALS", "BARE_VERBS_TRANSITIVE", "BARE_VERBS_ACTIVITY", "MENTAL_VERBS", "LINKING_VERBS", "ING_FORMS"],
    "NOUN": ["NAMES", "CREATURES", "OBJECTS", "PLACES", "FAMILY_AND_BODY", "MASS_AND_PLURAL_NOUNS", "THINGS_OUTDOORS",
             "THINGS_MISC", "ABSTRACT_NOUNS", "AFFECTION_NOUNS", "OBJECT_PRONOUNS", "NAMES_UNSPACED", "TIME_WORDS"],
    "PIECE": ["FRAGMENTS_SHORT", "FRAGMENTS_VC", "FRAGMENTS_ENDINGS", "FRAGMENTS_MIXED", "FRAGMENTS_D", "FRAGMENTS_G",
              "FRAGMENTS_H", "FRAGMENTS_L", "SUFFIXES_ING_ED", "SUFFIXES_PLURAL", "NAME_ENDINGS", "LETTERS_LOWER",
              "ONSETS_SINGLE", "ONSETS_A", "ONSETS_B", "ONSETS_C", "ONSETS_D", "VERB_ONSETS"],
    "FUNCTION": ["DETERMINERS", "DETERMINERS_RARE", "PREPOSITIONS", "SUBJECT_PRONOUNS", "ADVERBS", "PARTICLES"],
}
GNAMES = list(GROUPS) + ["OTHER"]
DIST_NEAR = np.array([0.07, -0.03, -0.37, -0.98])
R1, R2 = 0.129, 0.024
SELF = dict(PIECE=1.46, FUNCTION=0.85, VERB=0.51, OTHER=0.10, LINKER=-0.25, NOUN=-0.84, BOUNDARY=-1.47)
PREV = dict(LINKER=1.08, PIECE=0.92, OTHER=0.45, VERB=0.30, FUNCTION=0.13, NOUN=-0.56, BOUNDARY=-1.38)
KEY = dict(BOUNDARY=0.80, LINKER=0.01, VERB=0.00, FUNCTION=-0.48, OTHER=-0.48, NOUN=-0.50, PIECE=-1.03)

_G = None


def group_of():
    global _G
    if _G is None:
        from program.categories import ATTENTION_CLASSES
        from program.vocab import ID
        g = np.full(2000, len(GNAMES) - 1)
        for gi, gname in enumerate(GROUPS):
            for c in GROUPS[gname]:
                g[[ID[t] for t in ATTENTION_CLASSES[c]]] = gi
        _G = g
    return _G


def dist_score(d):
    dc = np.maximum(d, 0).astype(float)
    near = DIST_NEAR[np.clip(d, 0, 3)]
    mid = DIST_NEAR[3] - R1 * (np.minimum(dc, 32) - 3)
    far = -R2 * np.maximum(dc - 32, 0)
    return np.where(d <= 3, near, mid + far)


def attention(tokens):
    t = np.asarray(tokens)
    n = len(t)
    g = group_of()[t]
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    selfv = np.array([SELF[x] for x in GNAMES])
    prevv = np.array([PREV[x] for x in GNAMES])
    keyv = np.array([KEY[x] for x in GNAMES])
    s = dist_score(d) + (d == 0) * selfv[g][:, None] + (d == 1) * prevv[g][:, None] + keyv[g][None, :]
    s = np.where(d >= 0, s, -np.inf)
    s = s - s.max(1, keepdims=True)
    a = np.exp(s)
    return a / a.sum(1, keepdims=True)
