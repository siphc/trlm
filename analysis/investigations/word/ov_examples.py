"""Open-track insight: top votes of heads 0, 2, 3 when they read some word starts, next to corpus statistics."""
import numpy as np
import torch

from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.effects2 import pmi_table
from analysis.mech_eval import program
from harness.common import load_model
from program.vocab import ID, TOKENS

P = program()
s = P.STATS
m = load_model("attn_s0")
D, Dh = 128, 32
with torch.no_grad():
    E = m.wte.weight.double()
    pos = m.wpe.weight.double()[1:40].mean(0)
    blk = m.blocks[0]
    x = torch.nn.functional.layer_norm(E + pos, (D,), weight=blk.ln1.weight.double())
    Wv = blk.attn.qkv.weight.double()[2 * D:]
    Wo = blk.attn.out.weight.double()
    gf = m.ln_f.weight.double()
    U = {}
    for h in range(4):
        o = (x @ Wv[h * Dh:(h + 1) * Dh].T) @ Wo[:, h * Dh:(h + 1) * Dh].T
        o = o - o.mean(1, keepdim=True)
        U[h] = ((o * gf) @ E.T).numpy()
B = np.asarray(s.bigram_counts)
Bp = B / np.maximum(B.sum(1, keepdims=True), 1)
P1 = np.exp(np.asarray(s.log_follow, np.float64))
W2 = np.load(f"{WS.OUT}/W2.npy").astype(np.float64)
W2p = (W2 + 44 * s.p_uni) / (W2.sum(1, keepdims=True) + 44)
cond2 = np.log(W2p) - np.log(Bp @ P1 + 1e-12)
WE = pmi_table("WEND", s.p_uni)
show = lambda vec, k=8: "  ".join(f"{TOKENS[j]!r}" for j in np.argsort(-vec)[:k])
for w in [" app", " exc", " advent", " sh", " gr", " pr", " fr", " beaut", " Lil", " c"]:
    if w not in ID:
        continue
    t = ID[w]
    print(f"=== {w!r}  cont {s.cont[t]:.2f}; bigram: {show(np.log(Bp[t] + 1e-12), 6)}")
    for h in [0, 2, 3]:
        print(f"   head{h} vote : {show(U[h][t])}")
    print(f"   PMI_2      : {show(s.pmi[2][t])}")
    print(f"   cond2      : {show(cond2[t])}")
    print(f"   WEND       : {show(WE[t])}")
