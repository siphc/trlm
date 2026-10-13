"""RECALL with attention variant 'topical' (analysis/investigations/recall/fit_attn.py; fitted to head 1's own
attention pattern). Replaces the 15 RECALL vote directions with the same directions under the new attention."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import cand_common as C  # noqa: E402

NAME = "recall_attn_topical"
HEADS = [1]
REPLACES = C.BASE_KEYS


def features(P, tokens):
    return C.recall_features(P, tokens, "topical")[1]
