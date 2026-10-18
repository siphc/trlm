import numpy as np
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
P = program(); s = P.STATS
W = train_windows(16, seed=EVAL_SEED)
lp = model_logprobs(W)
C = np.asarray(s.bigram_counts)
mm, mq, kl_unseen, kl_seen, rare_m, rare_q = [], [], [], [], [], []
for w, p in zip(W, lp):
    t = np.asarray(w[:256], np.int64)
    q = P.predict(t)
    seen = C[t] > 0
    rare = (C[t] > 0) & (C[t] < 5)
    pm, pq = np.exp(p), np.exp(q)
    mm.append((pm * ~seen).sum(1)); mq.append((pq * ~seen).sum(1))
    rare_m.append((pm * rare).sum(1)); rare_q.append((pq * rare).sum(1))
    term = pm * (p - q)
    kl_unseen.append((term * ~seen).sum(1)); kl_seen.append((term * seen).sum(1))
f = lambda x: float(np.concatenate(x).mean())
print("model mass on corpus-unseen bigram continuations", f(mm), " program", f(mq))
print("model mass on rare (1-4 count) continuations", f(rare_m), " program", f(rare_q))
print("KL terms on unseen", f(kl_unseen), " on seen", f(kl_seen))
