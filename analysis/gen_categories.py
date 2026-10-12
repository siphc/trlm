"""Generate program/categories.py: every token assigned to one named word class.

Classes come from analysis/taxonomy.py (Ward clustering of TRAIN-split bigram contexts, no model
involved). Names were assigned by hand after reading each cluster; NAMES maps a cluster's most
frequent token to its name. Clusters whose head is a byte fragment or rare symbol are pooled, and
tokens that never occur in the train split form their own class.
Usage: python -m analysis.gen_categories
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "program"))
from vocab import ID, TOKENS  # noqa: E402

NAMES = {
    " and": "LINKERS", "\n": "SENTENCE_STARTS", ".": "SENTENCE_ENDS", " the": "DETERMINERS",
    " was": "LINKING_VERBS", " he": "SUBJECT_PRONOUNS", " so": "ADVERBS", " p": "ONSETS_SINGLE",
    " up": "PARTICLES", " it": "OBJECT_PRONOUNS", " Lily": "NAMES", "er": "FRAGMENTS_SHORT",
    " be": "BARE_VERBS_TRANSITIVE", "ar": "FRAGMENTS_VC", "le": "FRAGMENTS_ENDINGS",
    " had": "PAST_VERBS_MOTION", " saw": "PAST_VERBS_TRANSITIVE", " play": "BARE_VERBS_ACTIVITY",
    " mom": "FAMILY_AND_BODY", " in": "PREPOSITIONS", "ing": "SUFFIXES_ING_ED", "Once": "FRAGMENTS_MIXED",
    "s": "SUFFIXES_PLURAL", " little": "SIZE_AND_COLOUR", " day": "TIME_WORDS",
    " girl": "CREATURES", " fun": "MASS_AND_PLURAL_NOUNS", "The": "OPENERS_UNSPACED",
    " said": "SPEECH_VERBS", "p": "LETTERS_LOWER", " happy": "FEELINGS",
    " wanted": "VERBS_BEFORE_TO", " good": "QUALITIES", " tree": "OBJECTS", " park": "PLACES",
    " can": "MODALS", "Lily": "NAMES_UNSPACED", " an": "DETERMINERS_RARE", " M": "CAPITALS_SPACED",
    " re": "ONSETS_A", " playing": "ING_FORMS", " co": "ONSETS_B", " like": "MENTAL_VERBS",
    "I": "QUOTE_OPENERS", " sa": "ONSETS_C", " other": "MODIFIERS", " car": "THINGS_OUTDOORS",
    "se": "FRAGMENTS_D", " proud": "ABSTRACT_NOUNS", " k": "VERB_ONSETS", " sound": "THINGS_MISC",
    "ie": "NAME_ENDINGS", "w": "FRAGMENTS_G", "M": "CAPITALS_UNSPACED", "ver": "FRAGMENTS_H",
    " love": "AFFECTION_NOUNS", " didn": "NEGATED_AUXILIARIES", "ine": "FRAGMENTS_H",
    "Yes": "INTERJECTIONS", "age": "FRAGMENTS_H", "ist": "FRAGMENTS_H", " pr": "ONSETS_D",
    "our": "FRAGMENTS_L", " j": "ONSETS_D", "ick": "FRAGMENTS_L", " goodbye": "ABSTRACT_NOUNS",
    "and": "FRAGMENTS_L", " bus": "FRAGMENTS_L", " poin": "ONSETS_A", " ta": "ONSETS_A",
    "x": "FRAGMENTS_L", "'": "DIGITS_AND_SYMBOLS",
}
UNSEEN = "NEVER_IN_TRAINING_TEXT"
POOL = "BYTES_AND_RARE_SYMBOLS"
MAX = 64


def main() -> None:
    raw = json.load(open(os.path.join(REPO, "analysis", "out", "taxonomy_raw.json")))
    counts = np.load(os.path.join(REPO, "data", "baselines", "bigram_counts.npy")).sum(0)
    unseen = {TOKENS[i] for i in range(len(TOKENS)) if counts[i] == 0 and i != 0}
    classes: dict[str, list[str]] = {}
    pool: list[str] = []
    for c in raw:
        toks = [t for t in c["tokens"] if t not in unseen]
        name = NAMES.get(c["tokens"][0])
        if name is None:
            pool += toks
        else:
            classes.setdefault(name, []).extend(toks)
    for i in range(0, len(pool), MAX):
        classes[f"{POOL}_{i // MAX + 1}"] = pool[i : i + MAX]
    u = sorted(unseen, key=lambda t: ID[t])
    for i in range(0, len(u), MAX):
        classes[f"{UNSEEN}_{i // MAX + 1}"] = u[i : i + MAX]
    assert sorted(t for v in classes.values() for t in v) == sorted(TOKENS)
    assert all(len(v) <= MAX for v in classes.values()), {k: len(v) for k, v in classes.items() if len(v) > MAX}
    assert len(classes) <= MAX, len(classes)  # class-by-class tables must fit one literal per row
    lines = ['"""Word classes: every token belongs to exactly one named class.',
             "",
             "Found by Ward clustering of each token's left and right neighbours in the TRAIN split",
             "(analysis/taxonomy.py; no model involved), then named by reading the members",
             "(analysis/gen_categories.py). Each class has at most 64 tokens so that per-token tables",
             "can be written one class at a time. Members are listed most frequent first.",
             '"""', "", "from .vocab import ID", "", "CLASSES = {"]
    for name, toks in classes.items():
        lines.append(f"    {name!r}: [")
        row = []
        for t in toks:
            row.append(repr(t))
            if sum(len(r) + 2 for r in row) > 84:
                lines.append("        " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("        " + ", ".join(row) + ",")
        lines.append("    ],")
    lines += ["}", "", "CLASS_OF = {ID[t]: name for name, toks in CLASSES.items() for t in toks}",
              "CLASS_NAMES = list(CLASSES)", ""]
    with open(os.path.join(REPO, "program", "categories.py"), "w") as f:
        f.write("\n".join(lines))
    print(len(classes), "classes;", {k: len(v) for k, v in classes.items()})


if __name__ == "__main__":
    main()
