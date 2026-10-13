"""Step 10: how head 3's earlier-token effect relates to heads 0 and 2's (16 TRAIN windows, eval seed).
Sum of squares (model-probability weighted, centred per position) of each head set's effect, and the pooled
weighted correlation between head 3's effect and heads 0+2's effect, overall and by query group."""
import json
import numpy as np
from h3lib import OUT
import h3rule
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs
from harness.common import head_means, load_model

W = train_windows(16, seed=EVAL_SEED)
lp = model_logprobs(W)
means = head_means(load_model("attn_s0"), W)
eff = {}
for hs in ([3], [0, 2], [0, 2, 3]):
    eff[str(hs)] = lp - model_logprobs(W, ablate_heads=hs, ablation="earlier", means=means)
w = np.exp(lp)
def c(y):
    return y - (w * y).sum(-1, keepdims=True)
y3, y02, y023 = c(eff["[3]"]), c(eff["[0, 2]"]), c(eff["[0, 2, 3]"])
G = h3rule.group_of()[np.asarray(W[:, :256])]
res = {}
for g in ["all"] + h3rule.GNAMES:
    m = np.ones(G.shape, bool) if g == "all" else G == h3rule.GNAMES.index(g)
    ss = lambda y: float((w[m] * y[m] ** 2).sum())
    cov = float((w[m] * y3[m] * y02[m]).sum())
    res[g] = {"ss3": ss(y3), "ss02": ss(y02), "ss023": ss(y023), "corr_3_02": cov / np.sqrt(ss(y3) * ss(y02)),
              "nonadditivity": ss(y023 - y3 - y02) / ss(y023)}
    print(g, {k: round(v, 3) for k, v in res[g].items()}, flush=True)
json.dump(res, open(f"{OUT}/team.json", "w"), indent=1)
