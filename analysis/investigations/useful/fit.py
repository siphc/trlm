"""Fit the usefulness attention rules (rules.py) to model heads 0 and 2 of attn_s0 (pattern cross-entropy).

Fit windows: train_windows(32, 2024) (the program's attention fit windows); held-out pattern KL: train_windows(16, 2025).
Arms per head: program rule (as is), + usefulness with only USE_NEAR/USE_FAR fitted, + usefulness with every rule
constant refitted. Writes analysis/out/candidates/useful/fit_h{0,2}.json.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import SPECS, TABLES, Head0Use, Head2Use  # noqa: E402

from analysis.attention_fit import fit_rule, model_attention, pattern_ce  # noqa: E402
from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402


def main():
    import program.head0 as h0
    import program.head2 as h2

    P = program()
    s = P.STATS
    Wf, Wh = train_windows(32, seed=2024), train_windows(16, seed=2025)
    Af, Ah = model_attention("attn_s0", Wf), model_attention("attn_s0", Wh)
    for h, cls, mod in ((0, Head0Use, h0), (2, Head2Use, h2)):
        rule = cls()
        same = max(float(np.abs(rule.attention(np.asarray(w[:256], np.int64), s)
                                - mod.attention(np.asarray(w[:256], np.int64), s)).max()) for w in Wf[:4])
        res = {"check_max_abs_diff_vs_program_rule": same,
               "program rule": {"pattern_kl_heldout": float(pattern_ce(rule, Ah[:, h], Wh, s)[1])}}
        only = cls()
        f_only = fit_rule(only, Af[:, h], Wf, s, keep=True, spec=[("USE_NEAR", None, None), ("USE_FAR", None, None)])
        res["+ usefulness (2 scalars)"] = {"pattern_kl_heldout": float(pattern_ce(only, Ah[:, h], Wh, s)[1]),
                                          "fitted": f_only}
        full = cls()
        f_full = fit_rule(full, Af[:, h], Wf, s, keep=True, spec=SPECS[f"head{h}"])
        res["+ usefulness, rule refitted"] = {"pattern_kl_heldout": float(pattern_ce(full, Ah[:, h], Wh, s)[1]),
                                             "fitted": f_full}
        refit = cls()
        f_refit = fit_rule(refit, Af[:, h], Wf, s, keep=True, spec=SPECS[f"head{h}"][:-2])
        res["rule refitted, no usefulness (control)"] = {
            "pattern_kl_heldout": float(pattern_ce(refit, Ah[:, h], Wh, s)[1]), "fitted": f_refit}
        print(f"head {h}:", json.dumps({k: (v if not isinstance(v, dict) else v["pattern_kl_heldout"])
                                        for k, v in res.items()}), flush=True)
        with open(os.path.join(TABLES, f"fit_h{h}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
