"""Head 1 at onset queries: attention mass on earlier keys that are likely successors of the current token, vs the rule;
and a query-class-specific successor coefficient fitted to head 1's own attention."""
import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.h1_attn import collect, pattern_kl, FIT_SEED
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model

P = program()
PIECE = WS.piece_flags()
model = load_model("attn_s0")
fitd = collect(P, model, train_windows(32, seed=FIT_SEED), PIECE)
evald = collect(P, model, train_windows(32, seed=EVAL_SEED), PIECE)
P1 = np.exp(np.asarray(P.STATS.log_follow, np.float64))

# mass on successor keys at onset queries
acc = {}
for (A, base, succ, ind, cl), w in zip(evald, train_windows(32, seed=EVAL_SEED)):
    t = np.asarray(w[:256])
    n = len(t)
    sc = np.where(np.tril(np.ones((n, n), bool)), base, -np.inf)
    R = np.exp(sc - sc.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
    for i in range(1, n):
        c = cl[i]
        if c not in ("onset", "inside", "other"):
            continue
        pk = P1[t[i], t[:i]]
        for lab, m in (("P1>0.1", pk > 0.1), ("P1 0.01-0.1", (pk > 0.01) & (pk <= 0.1)), ("induction", ind[i, :i] > 0)):
            r = acc.setdefault((c, lab), [0, 0.0, 0.0, 0])
            r[0] += 1; r[1] += A[i, :i][m].sum(); r[2] += R[i, :i][m].sum(); r[3] += m.any()
for k, r in sorted(acc.items()):
    print(f"{k[0]:8s} keys {k[1]:12s}: head mass {r[1] / r[0]:.3f}  rule mass {r[2] / r[0]:.3f}  (queries with such a key {r[3] / r[0]:.2f})")

# class-specific successor coefficient, fitted to head 1's attention
def fit_cls(data, feat="pmi"):
    b = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    mask = torch.as_tensor(np.tril(np.ones((256, 256), bool)))
    tens = []
    for A, base, succ, ind, cl in data:
        isons = torch.as_tensor(np.array([c == "onset" for c in cl]))[:, None].double()
        isin = torch.as_tensor(np.array([c == "inside" for c in cl]))[:, None].double()
        tens.append((torch.as_tensor(A), torch.as_tensor(base), torch.as_tensor(succ), isons, isin))
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = 0.0
        for A, base, succ, o, ins in tens:
            sc = base + succ * (b[0] * (1 - o - ins) + b[1] * o + b[2] * ins)
            sc = sc.masked_fill(~mask, float("-inf"))
            loss = loss - (A * torch.log_softmax(sc, 1).masked_fill(~mask, 0)).sum() / (256 * len(tens))
        loss.backward()
        return loss
    opt.step(closure)
    return b.detach().numpy()

b = fit_cls(fitd)
print("class-specific B_SUCC (other, onset, inside):", np.round(b, 3))
ev = []
for A, base, succ, ind, cl in evald:
    o = np.array([c == "onset" for c in cl])[:, None]; ins = np.array([c == "inside" for c in cl])[:, None]
    ev.append((A, base + succ * (b[0] * (~o & ~ins) + b[1] * o + b[2] * ins), succ * 0, ind, cl))
print("pattern KL with class-specific successor:", {k: round(v, 3) for k, v in sorted(pattern_kl(ev, 0, 0).items())})
