import numpy as np, json
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.goal2.bottleneck import kl_rows
from harness.common import load_model
import torch
P = program(); s = P.STATS
W = train_windows(16, seed=EVAL_SEED)
model = load_model("attn_s0")
with torch.no_grad():
    logits = np.concatenate([model(torch.as_tensor(np.asarray(W[i:i+8, :256], np.int64))).double().numpy() for i in range(0, 16, 8)]).reshape(-1, 2000)
lp = logits - np.logaddexp.reduce(logits, 1, keepdims=True)
import program as Pk
score = []
for w in W:
    t = np.asarray(w[:256], np.int64)
    sc = np.zeros((256, 2000))
    for m in Pk.MECHANISMS:
        sc += m.votes(t, s)
    score.append(sc)
score = np.concatenate(score)
pu = s.unigram / s.unigram.sum()
def wstd(x, w):
    mu = (x * w).sum(1, keepdims=True)
    return np.sqrt((w * (x - mu) ** 2).sum(1))
r = {}
for name, wt in (("uniform", np.ones(2000) / 2000), ("unigram", pu)):
    a, b = wstd(logits, wt), wstd(score, wt)
    r[name] = {"model_mean": a.mean(), "model_cv": a.std() / a.mean(), "program_mean": b.mean(), "program_cv": b.std() / b.mean(),
               "corr": np.corrcoef(a, b)[0, 1]}
# rescale rule: score * (N0 / n)^gamma, n = weighted std of score
def lsm(x): return x - np.logaddexp.reduce(x, 1, keepdims=True)
base = kl_rows(lp, lsm(score)).mean()
r["base"] = base
best = None
for wname, wt in (("uniform", np.ones(2000) / 2000), ("unigram", pu)):
    n = wstd(score, wt); N0 = np.median(n)
    for g in (0.25, 0.5, 0.75, 1.0):
        for c in (0.9, 1.0, 1.1):
            k = kl_rows(lp, lsm(score * (c * N0 / n)[:, None] ** 1 * 0 + score * ((c * N0 / n) ** g)[:, None] - score)).mean()
            if best is None or k < best[0]: best = (k, wname, g, c)
r["best_rescale"] = best
# oracle: rescale to the model's own weighted std ratio
for wname, wt in (("uniform", np.ones(2000) / 2000), ("unigram", pu)):
    ratio = wstd(logits, wt) / wstd(score, wt)
    r[f"oracle_ratio_{wname}"] = kl_rows(lp, lsm(score * ratio[:, None])).mean()
print(json.dumps(r, indent=1, default=float))
