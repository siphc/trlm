"""LOCAL-0 with attention = program rule + usefulness of its own program vote (look-follows-write, Track B).
All head-0 vote directions are recomputed under the new attention (REPLACES the old ones)."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/candidates")
from useful_common import directions  # noqa: E402

NAME = "useful_h0"
HEADS = [0]
REPLACES = ["head0.self", "head0.near", "head0.far", "head0.earlier_successors", "head0.earlier_alike"]


def features(P, tokens):
    return directions(P, tokens, 0)
