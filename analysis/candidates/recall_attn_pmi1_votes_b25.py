"""RECALL with the pattern-fitted 'pmi1' attention term (score + 0.069 * PMI_1(t_i, t_j): prefer earlier words that
usually follow the current word) AND band-25..64 votes per word group. Combines the two near-miss candidates."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import cand_common as C  # noqa: E402

NAME = "recall_attn_pmi1_votes_b25"
HEADS = [1]
REPLACES = C.BASE_KEYS


def features(P, tokens):
    A, out = C.recall_features(P, tokens, "pmi1")
    out.update(C.per_group(P, tokens, A, C.b25_table(P), "b25"))
    return out
