"""RECALL with attention variant 'TT_exact' (analysis/investigations/recall/fit_attn.py; fitted to head 1's own
attention pattern). Replaces the 15 RECALL vote directions with the same directions under the new attention.
INSIGHT-ONLY CEILING: uses head 1's exact token-token score term from the weights. Never adoptable."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import cand_common as C  # noqa: E402

NAME = "recall_attn_TT_exact"
HEADS = [1]
REPLACES = C.BASE_KEYS


def features(P, tokens):
    return C.recall_features(P, tokens, "TT_exact")[1]
