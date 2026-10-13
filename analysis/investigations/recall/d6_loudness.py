"""D6: are the keys head 1 over-attends (relative to RECALL v1) quiet in what they write? (sinks)
OV loudness of token s = std over targets of head 1's OV row (exact weights, insight only); self = OV[s, s] - row mean.
Compared with the key features that improved the attention fit (topical, cap, quote, EOS)."""
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import fit_attn  # noqa: E402
import lib  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402
from program.vocab import TOKENS, ID  # noqa: E402

P = program()
OV = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/OV.npy").astype(np.float64)
uni = np.asarray(P.STATS.unigram, np.float64)
pv = uni / uni.sum()
mu = (OV * pv[None, :]).sum(1, keepdims=True)
loud = np.sqrt((((OV - mu) ** 2) * pv[None, :]).sum(1))      # probability-weighted std of the row
selfv = OV[np.arange(2000), np.arange(2000)] - mu[:, 0]
m = uni > 2000
w = uni[m] / uni[m].sum()
def wc(a, b):
    a, b = a[m] - (a[m] * w).sum(), b[m] - (b[m] * w).sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())
for nm in ["topical", "cap", "quote", "info8"]:
    f = fit_attn.key_flag(P, nm)
    print(f"{nm:8s} corr with loudness {wc(f, loud):+.3f}, with self vote {wc(f, selfv):+.3f}")
print("loudness / self vote by group (freq-weighted means):")
for k, g in enumerate(head1.GROUP_NAMES):
    mm = m & (head1.GROUP == k)
    ww = uni[mm] / uni[mm].sum()
    print(f"  {g:22s} loud {(loud[mm] * ww).sum():.2f} self {(selfv[mm] * ww).sum():.2f}")
for tk in ["<|eos|>", ' "', '"', '!"', '?"', '."', "Suddenly", "Once", " upon", " K", "K", " girl", " boy", " owl",
           " named", " She", " He", " the", " a", ".", "\n", ",", " Lily", " Tom", " ball", " was", " and"]:
    i = ID[tk]
    print(f"  {tk!r:12s} loud {loud[i]:.2f} self {selfv[i]:6.2f} topical {fit_attn.key_flag(P, 'topical')[i]:.3f} group {head1.GROUP_NAMES[head1.GROUP[i]]}")
