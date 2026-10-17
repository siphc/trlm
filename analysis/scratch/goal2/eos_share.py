import numpy as np
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from analysis.goal2.bottleneck import kl_rows
from program.vocab import TOKENS
P = program()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
toks = np.asarray(W[:, :256], np.int64)
K = np.stack([kl_rows(lp[i], P.predict(toks[i])) for i in range(64)])
tot = K.sum()
def rep(name, m):
    print(f"{name:45s} positions {m.mean()*100:5.2f}%  mean KL {K[m].mean():.3f}  share of total KL {K[m].sum()/tot*100:5.2f}%  excess over mean {(K[m]-K.mean()).sum()/K.size:.4f} nats/token")
pos = np.tile(np.arange(256), (64, 1))
after_eos = np.zeros_like(toks, bool); after_eos[:, 1:] = toks[:, :-1] == 0
after_eos2 = np.zeros_like(toks, bool); after_eos2[:, 2:] = toks[:, :-2] == 0
rep("position 0", pos == 0)
rep("positions 1-3", (pos >= 1) & (pos <= 3))
rep("current token = <eos>", toks == 0)
rep("first token after <eos>", after_eos)
rep("second token after <eos>", after_eos2)
rep("current = 'Once' after <eos>", after_eos & (toks == TOKENS.index("Once")))
