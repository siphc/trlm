"""LOCAL-2 with attention = program rule + usefulness of its own program vote (look-follows-write, Track B).
All head-2 vote directions are recomputed under the new attention (REPLACES the old ones)."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/candidates")
from useful_common import directions  # noqa: E402

NAME = "useful_h2"
HEADS = [2]
REPLACES = ["head2.self", "head2.near", "head2.far", "head2.no_repeat"]


def features(P, tokens):
    return directions(P, tokens, 2)
