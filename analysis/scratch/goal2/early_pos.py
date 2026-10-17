import numpy as np
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.goal2.bottleneck import kl_rows
P = program()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
Q = np.stack([P.predict(np.asarray(w[:256], np.int64)) for w in W])
H = lambda x: -(np.exp(x) * x).sum(-1)
Hm, Hq = H(lp), H(Q)
K = np.stack([kl_rows(lp[i], Q[i]) for i in range(64)])
for a, b in [(0,1),(1,2),(2,3),(3,4),(4,6),(6,8),(8,16),(16,64),(64,256)]:
    print(f"pos {a}-{b-1}: KL {K[:, a:b].mean():.3f}  H model {Hm[:, a:b].mean():.2f}  H program {Hq[:, a:b].mean():.2f}")
