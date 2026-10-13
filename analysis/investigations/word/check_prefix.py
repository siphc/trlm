import numpy as np, importlib.util
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M
from program.vocab import TOKENS
spec = importlib.util.spec_from_file_location("c", "analysis/candidates/word_prefix.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
P = program()
W = train_windows(16, seed=EVAL_SEED)
lp = model_logprobs(W)
res = {a: [] for a in [0, 0.1, 0.25, 0.5, 1.0]}
for w, p in zip(W, lp):
    t = np.asarray(w[:256])
    q = P.predict(t)
    f = c.features(P, t)
    F = f["L2"] + f["L3plus"]
    m = np.abs(F).sum(1) > 0
    for a in res:
        res[a].append(M.kl(p[m], M.log_softmax(q[m] + a * F[m])))
for a, v in res.items():
    print(a, np.concatenate(v).mean())
t = np.asarray(W[0][:256]); f = c.features(P, t); F = f["L2"] + f["L3plus"]
for i in np.where(np.abs(F).sum(1) > 0)[0][:4]:
    print(repr("".join(TOKENS[x] for x in t[i-3:i+1])), [ (TOKENS[j], round(F[i,j],2)) for j in np.argsort(-F[i])[:5]])
