"""Shared code for the useful_h* candidates: LOCAL-0 / LOCAL-2 vote directions under the usefulness attention rules
of analysis/investigations/useful/rules.py, with the constants fitted to the heads' own patterns (fit_h{0,2}.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/useful")
from rules import TABLES, Head0Use, Head2Use  # noqa: E402

_RULES = {}


def rule(h):
    if h not in _RULES:
        r = Head0Use() if h == 0 else Head2Use()
        fitted = json.load(open(os.path.join(TABLES, f"fit_h{h}.json")))["+ usefulness, rule refitted"]["fitted"]
        for key, v in fitted.items():
            attr, _, idx = key.partition("[")
            if idx:
                getattr(r, attr)[int(idx[:-1])] = v
            else:
                setattr(r, attr, v)
        _RULES[h] = r
    return _RULES[h]


def directions(P, tokens, h):
    from program.readers import earlier_only, mass_on_tokens, skipgram_votes

    s = P.STATS
    a = rule(h).attention(tokens, s)
    name = f"head{h}"
    out = {f"{name}.self": skipgram_votes(a, tokens, s, 1, 0, 0), f"{name}.near": skipgram_votes(a, tokens, s, 0, 1, 0),
           f"{name}.far": skipgram_votes(a, tokens, s, 0, 0, 1)}
    earlier = earlier_only(a)
    if h == 0:
        out["head0.earlier_successors"] = earlier @ np.asarray(s.pmi[1][tokens], np.float64)
        out["head0.earlier_alike"] = earlier @ np.asarray(s.slot_affinity[tokens], np.float64)
    else:
        out["head2.no_repeat"] = mass_on_tokens(earlier, tokens)
    return out
