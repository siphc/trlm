"""RECALL: recalled words also vote with extra corpus tables (b25), one strength per word group, under RECALL's
v1 attention (analysis/investigations/recall/cand_common.py has the table definitions)."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import cand_common as C  # noqa: E402
from program import head1  # noqa: E402

NAME = "recall_votes_b25"
HEADS = [1]
TABLES = "b25".split(",")


def features(P, tokens):
    A = head1.attention(tokens, P.STATS)
    out = {}
    for t in TABLES:
        out.update(C.per_group(P, tokens, A, getattr(C, f"{t}_table")(P), t))
    return out
