"""Generate program/categories.py: three class systems, in each of which every token has one named class.

ATTENTION classes come from analysis/taxonomy.py (Ward clustering of TRAIN-split bigram contexts, no model
involved). Names were assigned by hand after reading each cluster; NAMES maps a cluster's most
frequent token to its name. Clusters whose head is a byte fragment or rare symbol are pooled, and
tokens that never occur in the train split form their own class.
CONTEXT and TARGET classes come from analysis/cocluster.py (analysis/out/cocluster_classes.json).
Usage: python -m analysis.gen_categories   (python -m analysis.cocluster runs it after clustering)
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


COCLUSTER = os.path.join(REPO, "analysis", "out", "cocluster_classes.json")


def attention_classes() -> dict[str, list[str]]:
    """The Ward taxonomy (v1-v10's single class system; since the CONTEXT/TARGET split it serves the attention
    heads' PREFER tables only)."""
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
    assert len(classes) <= MAX, len(classes)  # PREFER rows must fit one literal
    return classes


def class_map(classes: dict[str, list[str]]) -> np.ndarray:
    """Class index (position in the dict) of every token id."""
    m = np.full(len(TOKENS), -1)
    for k, toks in enumerate(classes.values()):
        m[[ID[t] for t in toks]] = k
    assert (m >= 0).all()
    return m


def class_systems() -> dict[str, dict[str, list[str]]]:
    """CLASS_SYSTEMS of program/categories.py, read without importing the program package (importing it loads
    program/tables/, which may be mid-rewrite while a student is being emitted)."""
    import categories  # program/ is on sys.path

    return categories.CLASS_SYSTEMS


def class_maps() -> dict[str, np.ndarray]:
    """{system: class index of every token}, identical to program.tables_lib.TOKEN_CLASS."""
    return {name: class_map(classes) for name, classes in class_systems().items()}


def _block(var: str, classes: dict[str, list[str]]) -> list[str]:
    lines = [f"{var} = {{"]
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
    return lines + ["}", ""]


def write_categories(attention: dict, context: dict, target: dict) -> None:
    for name, classes in (("ATTENTION", attention), ("CONTEXT", context), ("TARGET", target)):
        assert sorted(t for v in classes.values() for t in v) == sorted(TOKENS), name
        assert all(len(v) <= MAX for v in classes.values()), (name, {k: len(v) for k, v in classes.items() if len(v) > MAX})
        assert all(k.isidentifier() for k in classes), name  # exception groups use class names as keywords
    assert len(target) <= MAX, len(target)  # each class-table row (one literal) has one number per TARGET class
    lines = ['"""Word classes. Three class systems; in each, every token belongs to exactly one named class.',
             "",
             f"ATTENTION_CLASSES  {len(attention)} classes, used by the attention heads' PREFER tables (which earlier tokens",
             "                   a head looks at, given the current token). Ward clustering of each token's left and",
             "                   right neighbours in the TRAIN split (analysis/taxonomy.py), named by hand",
             "                   (analysis/gen_categories.py).",
             f"CONTEXT_CLASSES    {len(context)} classes of tokens after which similar things come next: the rows of",
             "                   CLASS_FOLLOW and CLASS_RECALL (class of the current or of the attended token) and COPY.",
             f"TARGET_CLASSES     {len(target)} classes of tokens that similar contexts predict: the columns of CLASS_FOLLOW and",
             "                   CLASS_RECALL (class of the next token). Exceptions are grouped by them.",
             "",
             "CONTEXT and TARGET classes were found together by information-theoretic co-clustering of the TRAIN bigram",
             "counts (analysis/cocluster.py; no model involved): the row and column classes that keep the most",
             "information about which token comes next. Each table file in tables/ names the systems it uses.",
             "Each class has at most 64 tokens so that per-token tables can be written one class at a time.",
             "Classes and their members are listed most frequent first; names were given by reading the members.",
             '"""', ""]
    lines += _block("ATTENTION_CLASSES", attention) + _block("CONTEXT_CLASSES", context)
    lines += _block("TARGET_CLASSES", target)
    lines += ['CLASS_SYSTEMS = {"ATTENTION": ATTENTION_CLASSES, "CONTEXT": CONTEXT_CLASSES, "TARGET": TARGET_CLASSES}', ""]
    with open(os.path.join(REPO, "program", "categories.py"), "w") as f:
        f.write("\n".join(lines))
    print({k: len(v) for k, v in (("ATTENTION", attention), ("CONTEXT", context), ("TARGET", target))}, "classes")


def main() -> None:
    """Write program/categories.py from the Ward taxonomy and analysis/cocluster.py's CONTEXT/TARGET classes."""
    cc = json.load(open(COCLUSTER))
    write_categories(attention_classes(), cc["CONTEXT"], cc["TARGET"])


if __name__ == "__main__":
    main()
