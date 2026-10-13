"""Step 9: does head 3 share the program's phrase/local work with heads 0 and 2? R^2 of the earlier-token effect of
head sets {3}, {0,2}, {0,2,3}, {0,1,2,3} on the program parts (analysis/vote_fit.head_r2; 16 TRAIN windows, eval seed)."""
import json
import types
import numpy as np
from h3lib import OUT
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.vote_fit import head_r2

P = program()
W = train_windows(16, seed=EVAL_SEED)
lp = model_logprobs(W)
dummy = types.SimpleNamespace(NAME="none", features=lambda P, t: {"zero": np.zeros((len(t), 2000))})
res = {}
for hs in ([3], [0, 2], [0, 2, 3], [0, 1, 2, 3], [0], [2], [0, 3], [2, 3]):
    r = head_r2(P, W, lp, dummy, hs)
    res[str(hs)] = round(r["program"], 4)
    print(hs, res[str(hs)], flush=True)
json.dump(res, open(f"{OUT}/shared_r2.json", "w"), indent=1)
