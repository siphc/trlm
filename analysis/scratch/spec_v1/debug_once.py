"""Debug: per-mechanism contribution to the worst '<|eos|>Once' -> ' upon' positions of the v1 ledger (L4)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/scratch/spec_v1")
import v1probe as vp
from v1probe import Prog, T, kl_t, model_logprobs, train_windows, window_inputs
from program.vocab import TOKENS, ID
torch.set_num_threads(2)
wins = train_windows(160, seed=2025)
x = wins[:, :T]
once, upon = ID["Once"], ID[" upon"]
hits = [(b, i) for b in range(len(x)) for i in range(1, T) if x[b, i] == once and x[b, i - 1] == 0]
print("positions '<eos>Once':", len(hits))
b0 = sorted(set(b for b, _ in hits))[:6]
w = wins[b0]
pe = torch.as_tensor(model_logprobs(w))
ie = window_inputs(w, loo=True, extras=False)
# rebuild the L4 program from the ledger fit
import v1ablate  # noqa
H = torch.load(f"{vp.HERE}/v1probe_fitted.pt")[vp.MAIN]
led = json.load(open(f"{vp.HERE}/v1ledger_results.json"))
ab = json.load(open(f"{vp.HERE}/v1ablate_results.json"))
prog = Prog(); prog.load_state_dict(H)
for k, v in ab["stage1"]["params"].items(): prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
prog.p["h1_bic"].data.zero_()
for k, v in led["h1_attention_without_bic"].items(): prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
for k, v in led["ladder"]["L4 + PHRASE (trigram, 4-gram)"]["params"].items(): prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
prog.parts = {"h0", "h2", "recall", "phrase"}
with torch.no_grad():
    p = prog.p
    parts = {"follow": p["U"] * torch.as_tensor(vp.LU) + p["B"] * ie["BIG"], "h0": prog.local_votes("h0", ie),
             "h2": prog.local_votes("h2", ie), "recall": prog.recall_votes(ie),
             "phrase": p["TRI"] * ie["TRI"] + p["QUAD"] * ie["QUAD"]}
    tot = sum(parts.values())
    lq = torch.log_softmax(tot, -1)
for k, (b, i) in enumerate(hits):
    if b not in b0: continue
    bb = b0.index(b)
    kl = float(kl_t(pe[bb, i], lq[bb, i]))
    top = int(lq[bb, i].argmax())
    print(f"window {b} pos {i}: KL {kl:.2f}; model p(upon) {pe[bb, i, upon].exp():.3f}; program p(upon) {lq[bb, i, upon].exp():.4f}; program top {TOKENS[top]!r} {lq[bb, i, top].exp():.3f}")
    for nm, s in parts.items():
        c = s[bb, i]
        print(f"    {nm:7s} score(upon) - mean {float(c[upon] - c.mean()):+7.2f}  score(top) - mean {float(c[top] - c.mean()):+7.2f}")
    print("    tri feature(upon)", float(ie["TRI"][bb, i, upon]), " quad(upon)", float(ie["QUAD"][bb, i, upon]), " BIG(upon)", float(ie["BIG"][bb, i, upon]))
