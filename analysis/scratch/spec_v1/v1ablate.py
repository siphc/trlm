"""Follow-up to v1probe.py: a mechanistically faithful v1 and what each part earns.

Stage 1 (attention = the heads' attention): each reader's attention scalars are fitted to the MODEL's attention
pattern of its head (cross-entropy of the head's rows under the rule) on the 32 FIT windows, from the card values.
Stage 2 (votes): with attention frozen, only vote strengths are fitted to the model's log-probs (KL).
Then:
  - simplifications on the frozen base (each refits strengths only; attention rules that lose a term are re-fitted
    to the head's pattern first): no in-context successor term, no successor terms at all, recall votes tied
    across groups, topic-only recall, no 4-gram, no head-0 / head-3 reader, no echo;
  - penalised joint fits: KL_out + lam * sum_r patternKL_r, lam in LAMS: the trade-off between fidelity and
    mechanistic match;
  - consistency (AGENTS.md section 9): program KL rise when a part is removed, next to the model's KL rise when the
    matching head (or its earlier-token part) is mean-ablated, on the same windows.
Held-out TRAIN windows (seed 2025), paired per-window differences.

Run after v1probe.py: OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python analysis/scratch/spec_v1/v1ablate.py [n_eval]
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.cpu_count = lambda: 2          # harness.model_logprobs sets torch threads to min(8, cpu_count): keep it at 2
import numpy as np
import torch

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/scratch/spec_v1")
import v1probe as vp  # noqa: E402
from v1probe import (ATTN_KEYS, LOC, MAIN, P0, REC, Prog, T, kl_t, log, model_logprobs, model_patterns,  # noqa: E402
                     sub, train_windows, window_inputs)

torch.set_num_threads(2)
HERE = vp.HERE
HEAD_OF = {"h0": 0, "h1": 1, "h2": 2, "h3": 3}
VOTES = LOC + ["TRI", "QUAD", "ECHO"] + REC
FULL = {"h0", "h2", "h3", "phrase", "recall", "echo"}
LAMS = [0.01, 0.03, 0.1]


class TiedProg(Prog):
    """Recall votes shared by all key groups (element 0 of h1_copy / h1_topic / h1_next)."""

    def recall_votes(self, inp):
        p, x = self.p, inp["x"]
        b = self.attn("h1", inp)
        onehot = torch.nn.functional.one_hot(x, vp.V).float()
        src = p["h1_copy"][0] * onehot + p["h1_topic"][0] * inp["FAR"] + p["h1_next"][0] * inp["G"][:, 0]
        return torch.matmul(b, src)


def pattern_ce(prog, r, inp, Am, sl):
    """mean over query rows of -sum_j A_model(i,j) log a_rule(i,j) (= pattern KL + a constant)."""
    ls = torch.log_softmax(prog.scores(r, sub(inp, sl)), -1)
    A = Am[sl, HEAD_OF[r]]
    return -(A * torch.where(A > 0, ls, torch.zeros_like(ls))).sum(-1).mean()


def fit_patterns(prog, r, inp, Am, keys, iters=100, chunk=8):
    for k, prm in prog.p.items():
        prm.requires_grad_(k in keys)
    opt = torch.optim.LBFGS([prog.p[k] for k in keys], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    n = len(Am)

    def closure():
        opt.zero_grad()
        tot = 0.0
        for c in range(0, n, chunk):
            loss = pattern_ce(prog, r, inp, Am, slice(c, c + chunk)) * min(chunk, n - c) / n
            loss.backward()
            tot += loss.item()
        return torch.tensor(tot)

    opt.step(closure)


def fit_joint(prog, inp, plog, Am, free, lam=0.0, readers=(), iters=60, chunk=8):
    """KL(model || program) + lam * sum over readers of pattern cross-entropy against the matching head."""
    for k, prm in prog.p.items():
        prm.requires_grad_(k in free)
    opt = torch.optim.LBFGS([prog.p[k] for k in free], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20)
    n = len(plog)

    def closure():
        opt.zero_grad()
        tot = 0.0
        for c in range(0, n, chunk):
            sl = slice(c, c + chunk)
            loss = kl_t(plog[sl], prog(sub(inp, sl))).sum() / (n * T)
            if lam:
                for r in readers:
                    loss = loss + lam * pattern_ce(prog, r, inp, Am, sl) * min(chunk, n - c) / n
            loss.backward()
            tot += loss.item()
        return torch.tensor(tot)

    opt.step(closure)
    with torch.no_grad():
        return sum(float(kl_t(plog[c:c + chunk], prog(sub(inp, slice(c, c + chunk)))).sum())
                   for c in range(0, n, chunk)) / (n * T)


def main():
    from harness.common import head_means, load_model
    from harness.common import model_logprobs as harness_logprobs

    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    smoke = os.environ.get("SMOKE") == "1"
    nfit = 4 if smoke else 32
    iters = 2 if smoke else 60
    fitw = train_windows(32, seed=2024)[:nfit]
    plog = torch.as_tensor(model_logprobs(fitw))
    inp = window_inputs(fitw, loo=True, extras=False)
    Am = model_patterns(fitw)
    log("FIT inputs and model patterns ready")
    H = torch.load(f"{HERE}/v1probe_fitted.pt")[MAIN]
    out = {"stage1": {}, "variants": {}}

    # ---- stage 1: attention rules fitted to the heads' patterns
    base = Prog()
    base.load_state_dict(H)
    for k in sum(ATTN_KEYS.values(), []):        # restart attention from the card values
        base.p[k].data.copy_(torch.tensor(P0[k], dtype=torch.float32))
    with torch.no_grad():
        card = {r: float(sum(pattern_ce(base, r, inp, Am, slice(c, c + 8)) * min(8, nfit - c) / nfit
                             for c in range(0, nfit, 8))) for r in HEAD_OF}
    for r in HEAD_OF:
        fit_patterns(base, r, inp, Am, ATTN_KEYS[r], iters=3 if smoke else 100)
    with torch.no_grad():
        st1 = {r: float(sum(pattern_ce(base, r, inp, Am, slice(c, c + 8)) * min(8, nfit - c) / nfit
                            for c in range(0, nfit, 8))) for r in HEAD_OF}
    out["stage1"] = {"pattern_ce_card": card, "pattern_ce_stage1": st1,
                     "params": {k: np.round(base.p[k].detach().numpy(), 4).tolist() for k in sum(ATTN_KEYS.values(), [])}}
    log("stage 1 pattern CE card -> refit:", card, st1)

    def variant(name, cls=Prog, parts=FULL, free=VOTES, zero=(), refit_attn=None, lam=0.0, start=None):
        prog = cls()
        prog.load_state_dict(start if start is not None else base.state_dict())
        for z in zero:
            prog.p[z].data.zero_()
        if cls is TiedProg:
            for k in REC:
                prog.p[k].data[0] = prog.p[k].data.mean()
        if refit_attn:      # (reader, keys): attention rule re-fitted to its head's pattern without the zeroed terms
            fit_patterns(prog, refit_attn[0], inp, Am, refit_attn[1], iters=3 if smoke else 100)
        prog.parts = set(parts)
        readers = [r for r in HEAD_OF if r in parts or (r == "h1" and "recall" in parts)]
        tl = fit_joint(prog, inp, plog, Am, free, lam=lam, readers=readers, iters=iters)
        n_free = sum(prog.p[k].numel() for k in free) - (12 if cls is TiedProg else 0)
        out["variants"][name] = {"train_kl": tl, "n_free": int(n_free), "lam": lam,
                                 "params": {k: np.round(prog.p[k].detach().numpy(), 4).tolist() for k in free}}
        log(f"fit {name}: train KL {tl:.4f} ({n_free} free scalars)")
        return name, (cls, set(parts), {k: v.detach().clone() for k, v in prog.state_dict().items()})

    h1_no_ic = [k for k in ATTN_KEYS["h1"] if k != "h1_bic"]
    h1_no_succ = [k for k in ATTN_KEYS["h1"] if k not in ("h1_bic", "h1_bclass")]
    fitted = dict([
        variant("V1F frozen head attention (stage 1) + fitted votes"),
        variant("V1F without in-context successor term (b_ic = 0, h1 attention re-fitted)", zero=["h1_bic"],
                refit_attn=("h1", h1_no_ic)),
        variant("V1F without successor terms (b_ic = b_class = 0, h1 re-fitted)", zero=["h1_bic", "h1_bclass"],
                refit_attn=("h1", h1_no_succ)),
        variant("V1F recall votes tied across groups", cls=TiedProg),
        variant("V1F recall topic votes only (no copy, no next)", free=[k for k in VOTES if k not in ("h1_copy", "h1_next")],
                zero=["h1_copy", "h1_next"]),
        variant("V1F without 4-gram", free=[k for k in VOTES if k != "QUAD"], zero=["QUAD"]),
        variant("V1F without echo", parts=FULL - {"echo"}, free=[k for k in VOTES if k != "ECHO"], zero=["ECHO"]),
        variant("V1F without head-0 reader", parts=FULL - {"h0"}, free=[k for k in VOTES if k != "S0"]),
        variant("V1F without head-3 reader", parts=FULL - {"h3"}, free=[k for k in VOTES if k != "S3"]),
    ])
    allatt = sum(ATTN_KEYS.values(), [])
    for lam in LAMS:
        fitted.update([variant(f"V1R penalised joint fit lam={lam}", free=VOTES + allatt, lam=lam,
                               start=fitted["V1F frozen head attention (stage 1) + fitted votes"][2])])
    json.dump(out, open(f"{HERE}/v1ablate_results.json", "w"), indent=1)
    del inp, plog
    # ---- evaluation on held-out TRAIN windows
    evw = train_windows(160, seed=2025)[: (4 if smoke else n_eval)]
    n_ev = len(evw)
    per = {name: [] for name in fitted}
    pkl = {name: {r: [] for r in HEAD_OF} for name in fitted}
    removal = {k: [] for k in ["base", "h0 ctx", "h2 ctx", "h3 ctx", "all local ctx", "h0", "h2", "h3", "recall",
                               "phrase", "echo"]}
    model = load_model("attn_s0")
    n_model = min(n_ev, 64)
    means = head_means(model, np.asarray(evw[:n_model]))
    torch.set_num_threads(2)
    model_rise = {f"{kind} head {h}": [] for h in range(4) for kind in ("mean", "earlier")}
    ref = "V1F frozen head attention (stage 1) + fitted votes"
    for c in range(0, n_ev, 16):
        w = evw[c:c + 16]
        pe = torch.as_tensor(model_logprobs(w))
        ie = window_inputs(w, loo=True, extras=False)
        Ae = model_patterns(w[:4]) if c < 32 else None
        for name, (cls, parts, sd) in fitted.items():
            prog = cls()
            prog.load_state_dict(sd)
            prog.parts = parts
            with torch.no_grad():
                per[name].append(kl_t(pe, prog(ie)).mean(-1).numpy())
                if Ae is not None:
                    for r, h in HEAD_OF.items():
                        if r in parts or (r == "h1" and "recall" in parts):
                            pkl[name][r].append(vp.pattern_kl(Ae[:, h], prog.attn(r, sub(ie, slice(0, 4)))))
        cls, parts, sd = fitted[ref]
        prog = cls()
        prog.load_state_dict(sd)
        prog.parts = parts
        with torch.no_grad():
            removal["base"].append(kl_t(pe, prog(ie)).mean(-1).numpy())
            for r in ("h0", "h2", "h3"):
                removal[r].append(kl_t(pe, prog(ie, off=(r,))).mean(-1).numpy())
                S = prog.p["S" + r[1]].data.clone()
                prog.p["S" + r[1]].data[1:] = 0.0
                removal[f"{r} ctx"].append(kl_t(pe, prog(ie)).mean(-1).numpy())
                prog.p["S" + r[1]].data.copy_(S)
            saved = {r: prog.p["S" + r[1]].data.clone() for r in ("h0", "h2", "h3")}
            for r in saved:
                prog.p["S" + r[1]].data[1:] = 0.0
            removal["all local ctx"].append(kl_t(pe, prog(ie)).mean(-1).numpy())
            for r, S in saved.items():
                prog.p["S" + r[1]].data.copy_(S)
            for k in ("recall", "phrase", "echo"):
                removal[k].append(kl_t(pe, prog(ie, off=(k,))).mean(-1).numpy())
        if c < n_model:
            x = np.asarray(w)
            for h in range(4):
                for kind in ("mean", "earlier"):
                    lp = torch.as_tensor(harness_logprobs(model, x, ablate_heads=(h,), ablation=kind, means=means))
                    model_rise[f"{kind} head {h}"].append(kl_t(pe, lp.float()).mean(-1).numpy())
            torch.set_num_threads(2)
        log(f"eval {c + len(w)}/{n_ev}")
    ctrl = np.concatenate(per[ref])
    out["eval"] = {"n_windows": n_ev, "variants": {}}
    for name in fitted:
        k = np.concatenate(per[name])
        d = k - ctrl
        out["eval"]["variants"][name] = {"kl": float(k.mean()), "se": float(k.std(ddof=1) / np.sqrt(len(k))),
                                         "delta_vs_V1F": float(d.mean()),
                                         "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                         "pattern_kl": {r: float(np.mean(v)) for r, v in pkl[name].items() if v},
                                         "n_free": out["variants"][name]["n_free"]}
        e = out["eval"]["variants"][name]
        log(f"EVAL {name:78s} KL {e['kl']:.4f}  vs V1F {e['delta_vs_V1F']:+.4f} +- {e['delta_se']:.4f}  "
            f"({e['n_free']} scalars) pattern KL {e['pattern_kl']}")
    b = np.concatenate(removal["base"])
    out["eval"]["program_removal_rise_V1F"] = {k: float(np.concatenate(v).mean() - b.mean()) for k, v in removal.items()
                                               if k != "base"}
    out["eval"]["model_ablation_rise"] = {"n_windows": n_model,
                                          **{k: float(np.concatenate(v).mean()) for k, v in model_rise.items()}}
    log("program removal rise (V1F, no refit):", out["eval"]["program_removal_rise_V1F"])
    log("model ablation rise:", out["eval"]["model_ablation_rise"])
    json.dump(out, open(f"{HERE}/v1ablate_results.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
