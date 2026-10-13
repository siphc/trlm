"""D11: is the clipping of the class successor (max(0, PMI)) hiding signal? Unclipped class PMI
log P(TARGET class of k | CONTEXT class of q) - log P(TARGET class of k) vs head 1's TT interaction (as in D8)."""
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program.corpus import class_index  # noqa: E402

P = program()
s = P.STATS
uni = np.asarray(s.unigram, np.float64)
TT = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/TT.npy").astype(np.float64)
qs = np.argsort(-uni)[:800]
ks = np.argsort(-uni)[:1500]
wq = uni[qs] / uni[qs].sum()
wk = uni[ks] / uni[ks].sum()
W = np.outer(wq, wk)


def dc(M):
    M = M[np.ix_(qs, ks)].astype(np.float64)
    M = M - (M * wk[None, :]).sum(1, keepdims=True)
    return M - (M * wq[:, None]).sum(0, keepdims=True)


T0 = dc(TT)
sst = (W * T0 * T0).sum()
B = np.asarray(s.bigram_counts, np.float64)
ctx, tgt = class_index("CONTEXT"), class_index("TARGET")
nc, nt = ctx.max() + 1, tgt.max() + 1
Mc = np.zeros((nc, 2000)); Mc[ctx, np.arange(2000)] = 1
Mt = np.zeros((2000, nt)); Mt[np.arange(2000), tgt] = 1
ccc = Mc @ B @ Mt
p_k = (ccc.sum(0) + 1) / (ccc.sum() + nt)
pmi_cc = np.log((ccc + 1) / (ccc.sum(1, keepdims=True) + nt)) - np.log(p_k)[None, :]
qtok = B @ Mt                                         # query token x TARGET class of the next token
pmi_qt = np.log((qtok + 1) / (qtok.sum(1, keepdims=True) + nt)) - np.log(p_k)[None, :]
cands = {"class_successor (clipped, program)": np.asarray(s.class_successor, np.float64),
         "class PMI unclipped": pmi_cc[ctx][:, tgt],
         "class PMI clipped at -2": np.maximum(pmi_cc, -2)[ctx][:, tgt],
         "query TOKEN x key TARGET class PMI": pmi_qt[:, tgt],
         "query TOKEN x key TARGET class PPMI": np.maximum(pmi_qt, 0)[:, tgt]}
for nm, M in cands.items():
    C = dc(M)
    r = (W * T0 * C).sum() / np.sqrt(sst * (W * C * C).sum())
    print(f"{nm:40s} corr {r:+.3f} R2 {r * r:.3f}")
