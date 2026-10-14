"""LOCAL-0 and LOCAL-2 both with usefulness attention (look-follows-write, Track B)."""
import sys

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/candidates")
from useful_common import directions  # noqa: E402

NAME = "useful_h02"
HEADS = [0, 2]
REPLACES = ["head0.self", "head0.near", "head0.far", "head0.earlier_successors", "head0.earlier_alike",
            "head2.self", "head2.near", "head2.far", "head2.no_repeat"]


def features(P, tokens):
    return {**directions(P, tokens, 0), **directions(P, tokens, 2)}
