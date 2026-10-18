import numpy as np, json
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.program_gap.bottleneck import kl_rows
P = program()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
K = np.stack([kl_rows(lp[i], P.predict(np.asarray(W[i, :256], np.int64))) for i in range(64)])
m = K.mean(0)
print("mean", K.mean())
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]:
    print(f"pos {a}-{b-1}: KL {m[a:b].mean():.3f}  contributes {m[a:b].sum()/256:.4f}")
