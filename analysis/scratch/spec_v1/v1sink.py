"""Residual iteration test: the end-of-story token as an attention sink for RECALL (head 1).

Measured in the model (debug_boundary.py, 80 TRAIN windows): one token after an end-of-story token, head 1 puts
0.48 of its attention on that token (0.31 at 2-4 tokens after, 0.23 at 5-16, 0.09 at 17-64). The v1 recall rule
treats EOS as an ordinary FUNCTION_WORDS key, so it reads the previous story instead, e.g. '<eos>Once' -> ' there'.
Variant: RECALL score gets + SAL_EOS * [key is EOS] and EOS keys vote nothing (a sink). Head-1 attention re-fitted
to the head's pattern (stage 1), then votes refitted (stage 2), compared with the v1 ledger rung L4 (paired).

Run after v1ledger.py: OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python analysis/scratch/spec_v1/v1sink.py
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import numpy as np
import torch

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/scratch/spec_v1")
import v1probe as vp  # noqa: E402
from v1probe import ATTN_KEYS, REC, Prog, T, kl_t, log, model_logprobs, model_patterns, sub, train_windows, window_inputs  # noqa: E402
from v1ablate import fit_joint, fit_patterns  # noqa: E402

torch.set_num_threads(2)
HERE = vp.HERE
PARTS = {"h0", "h2", "recall", "phrase"}
VOTES = ["U", "B", "S0", "S2"] + REC + ["TRI", "QUAD"]
H1_ATT = [k for k in ATTN_KEYS["h1"] if k != "h1_bic"]


class SinkProg(Prog):
    def __init__(self):
        super().__init__()
        self.p["h1_sal_eos"] = torch.nn.Parameter(torch.tensor(0.0))

    def scores(self, r, inp):
        s = super().scores(r, inp)
        if r == "h1":
            s = s + self.p["h1_sal_eos"] * (inp["x"] == 0).float()[:, None, :]
        return s

    def recall_votes(self, inp):
        p, x = self.p, inp["x"]
        b = self.attn("h1", inp)
        g = vp.tok(vp.GRP1, x)
        keep = (x != 0).float()[:, :, None]
        onehot = torch.nn.functional.one_hot(x, vp.V).float()
        src = (p["h1_copy"][g][:, :, None] * onehot + p["h1_topic"][g][:, :, None] * inp["FAR"]
               + p["h1_next"][g][:, :, None] * inp["G"][:, 0])
        return torch.matmul(b, src * keep)


def load_l4(prog):
    H = torch.load(f"{HERE}/v1probe_fitted.pt")[vp.MAIN]
    led = json.load(open(f"{HERE}/v1ledger_results.json"))
    ab = json.load(open(f"{HERE}/v1ablate_results.json"))
    prog.load_state_dict(H, strict=False)
    for k, v in ab["stage1"]["params"].items():
        prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
    prog.p["h1_bic"].data.zero_()
    for k, v in led["h1_attention_without_bic"].items():
        prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
    for k, v in led["ladder"]["L4 + PHRASE (trigram, 4-gram)"]["params"].items():
        prog.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
    prog.parts = set(PARTS)
    return prog


def main():
    fitw = train_windows(32, seed=2024)
    plog = torch.as_tensor(model_logprobs(fitw))
    inp = window_inputs(fitw, loo=True, extras=False)
    Am = model_patterns(fitw)
    l4 = load_l4(Prog())
    sink = load_l4(SinkProg())
    fit_patterns(sink, "h1", inp, Am, H1_ATT + ["h1_sal_eos"], iters=100)
    tl = fit_joint(sink, inp, plog, Am, VOTES, iters=60)
    log(f"sink variant: SAL_EOS {float(sink.p['h1_sal_eos']):.3f}, train KL {tl:.4f}")
    out = {"h1_attention": {k: np.round(sink.p[k].detach().numpy(), 4).tolist() for k in H1_ATT + ["h1_sal_eos"]},
           "votes": {k: np.round(sink.p[k].detach().numpy(), 4).tolist() for k in VOTES}, "train_kl": tl}
    del inp, plog
    wins = train_windows(160, seed=2025)
    x = wins[:, :T]
    ii = np.arange(T)
    start = np.array([np.maximum.accumulate(np.where(r == 0, ii, -1)) for r in x])
    since = np.where(start >= 0, ii[None, :] - start, 10 ** 6)
    k_l4, k_s, pk_l4, pk_s = [], [], [], []
    for c in range(0, len(wins), 16):
        w = wins[c:c + 16]
        pe = torch.as_tensor(model_logprobs(w))
        ie = window_inputs(w, loo=True, extras=False)
        with torch.no_grad():
            k_l4.append(kl_t(pe, l4(ie)).numpy())
            k_s.append(kl_t(pe, sink(ie)).numpy())
            if c < 32:
                Ae = model_patterns(w[:4])
                pk_l4.append(vp.pattern_kl(Ae[:, 1], l4.attn("h1", sub(ie, slice(0, 4)))))
                pk_s.append(vp.pattern_kl(Ae[:, 1], sink.attn("h1", sub(ie, slice(0, 4)))))
        log(f"eval {c + len(w)}/{len(wins)}")
    a, b = np.concatenate(k_l4), np.concatenate(k_s)
    d = b.mean(1) - a.mean(1)
    out["eval"] = {"l4_kl": float(a.mean()), "sink_kl": float(b.mean()), "delta": float(d.mean()),
                   "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                   "h1_pattern_kl_l4": float(np.mean(pk_l4)), "h1_pattern_kl_sink": float(np.mean(pk_s))}
    for lo, hi in [(0, 0), (1, 1), (2, 4), (5, 16), (17, 64)]:
        m = (since >= lo) & (since <= hi)
        out["eval"][f"tokens since EOS {lo}-{hi}"] = {"share_positions": float(m.mean()), "l4": float(a[m].mean()),
                                                      "sink": float(b[m].mean())}
    log(json.dumps(out["eval"], indent=1))
    json.dump(out, open(f"{HERE}/v1sink_results.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
