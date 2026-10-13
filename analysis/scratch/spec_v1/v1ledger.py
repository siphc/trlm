"""Mechanism ledger for the recommended v1 (v1ablate.py results: no head-3 reader, no echo, no in-context successor
term): attention frozen at the heads' own patterns (stage 1; head 1 re-fitted to its pattern without the in-context
term); add the mechanisms in build order, refit only the vote strengths at each step; then the penalised joint fit
(stage 3, lam = 0.1); characterise the residual.

  L1 FOLLOW (current token -> corpus bigram)                      direct path + self parts of heads 0/2(/3)
  L2 + LOCAL readers (heads 0, 2; skip-gram votes)
  L3 + RECALL (head 1; co-occurrence / copy / next votes)
  L4 + PHRASE (trigram + 4-gram backoff)                          behavioural stand-in   = v1 (faithful)
  L3' = L2 + PHRASE (no RECALL), for the other order
  L5 = L4 with stage 3: KL + 0.1 * sum of pattern cross-entropies, all scalars free

Held-out TRAIN windows (seed 2025) and dev_fast (final rung only); per-position KL saved for the residual slices.
Run after v1ablate.py: OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python analysis/scratch/spec_v1/v1ledger.py
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
from v1probe import LOC, MAIN, REC, Prog, T, class_index, kl_t, log, model_logprobs, train_windows, window_inputs  # noqa: E402
from v1ablate import fit_joint, fit_patterns  # noqa: E402
from v1probe import ATTN_KEYS, model_patterns  # noqa: E402

torch.set_num_threads(2)
HERE = vp.HERE
LOC2 = ["U", "B", "S0", "S2"]
LADDER = [  # name, parts, free vote scalars
    ("L1 FOLLOW", set(), ["U", "B"]),
    ("L2 + LOCAL readers (heads 0, 2)", {"h0", "h2"}, LOC2),
    ("L3 + RECALL (head 1)", {"h0", "h2", "recall"}, LOC2 + REC),
    ("L4 + PHRASE (trigram, 4-gram)", {"h0", "h2", "recall", "phrase"}, LOC2 + REC + ["TRI", "QUAD"]),
    ("L3' L2 + PHRASE (no RECALL)", {"h0", "h2", "phrase"}, LOC2 + ["TRI", "QUAD"]),
]
H1_ATT = [k for k in ATTN_KEYS["h1"] if k != "h1_bic"]
STAGE3 = ("L5 = L4 + stage 3 (penalised joint fit, lam 0.1)", {"h0", "h2", "recall", "phrase"},
          LOC2 + REC + ["TRI", "QUAD"] + ATTN_KEYS["h0"] + ATTN_KEYS["h2"] + H1_ATT)


def main():
    from harness.common import eval_windows
    from program.vocab import TOKENS

    smoke = os.environ.get("SMOKE") == "1"
    ab = json.load(open(f"{HERE}/v1ablate_results.json"))
    H = torch.load(f"{HERE}/v1probe_fitted.pt")[MAIN]
    base = Prog()
    base.load_state_dict(H)
    for k, v in ab["stage1"]["params"].items():                 # frozen head attention (stage 1)
        base.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
    for k, v in ab["variants"]["V1F frozen head attention (stage 1) + fitted votes"]["params"].items():
        base.p[k].data.copy_(torch.tensor(v, dtype=torch.float32))
    fitw = train_windows(32, seed=2024)[: 4 if smoke else 32]
    plog = torch.as_tensor(model_logprobs(fitw))
    inp = window_inputs(fitw, loo=True, extras=False)
    Am = model_patterns(fitw)
    base.p["h1_bic"].data.zero_()                    # no in-context successor term; head 1 re-fitted to its pattern
    fit_patterns(base, "h1", inp, Am, H1_ATT, iters=3 if smoke else 100)
    fitted, out = {}, {"ladder": {}, "h1_attention_without_bic": {k: np.round(base.p[k].detach().numpy(), 4).tolist()
                                                                    for k in H1_ATT}}
    for name, parts, free in LADDER:
        prog = Prog()
        prog.load_state_dict(base.state_dict())
        prog.parts = set(parts)
        tl = fit_joint(prog, inp, plog, None, free, iters=2 if smoke else 60)
        fitted[name] = (set(parts), {k: v.detach().clone() for k, v in prog.state_dict().items()})
        out["ladder"][name] = {"train_kl": tl, "n_vote_scalars": int(sum(prog.p[k].numel() for k in free)),
                               "params": {k: np.round(prog.p[k].detach().numpy(), 4).tolist() for k in free}}
        log(f"fit {name}: train KL {tl:.4f}")
    name, parts, free = STAGE3
    prog = Prog()
    prog.load_state_dict(fitted["L4 + PHRASE (trigram, 4-gram)"][1])
    prog.parts = set(parts)
    tl = fit_joint(prog, inp, plog, Am, free, lam=0.1, readers=["h0", "h1", "h2"], iters=2 if smoke else 60)
    fitted[name] = (set(parts), {k: v.detach().clone() for k, v in prog.state_dict().items()})
    out["ladder"][name] = {"train_kl": tl, "n_scalars": int(sum(prog.p[k].numel() for k in free)),
                           "params": {k: np.round(prog.p[k].detach().numpy(), 4).tolist() for k in free}}
    log(f"fit {name}: train KL {tl:.4f}")
    LADDER.append(STAGE3)
    del inp, plog
    res = {}
    for split, wins, loo, names in [("train_eval", train_windows(160, seed=2025), True, [n for n, _, _ in LADDER]),
                                    ("dev_fast", eval_windows("dev_fast"), False,
                                     ["L1 FOLLOW", "L4 + PHRASE (trigram, 4-gram)", STAGE3[0]])]:
        wins = wins[:4] if smoke else wins
        per = {n: [] for n in names}
        top1 = {n: [] for n in names}
        plist = []
        qarg = {n: [] for n in names}
        pk = {n: {r: [] for r in ("h0", "h1", "h2")} for n in names}
        for c in range(0, len(wins), 16):
            w = wins[c:c + 16]
            pe = torch.as_tensor(model_logprobs(w))
            ie = window_inputs(w, loo=loo, extras=False)
            Ae = model_patterns(w[:4]) if (split == "train_eval" and c < 32) else None
            if split == "train_eval":
                plist.append(pe.max(-1).values.exp().reshape(-1).numpy())
                plist.append(pe.argmax(-1).reshape(-1).numpy())
            for n in names:
                prog = Prog()
                prog.load_state_dict(fitted[n][1])
                prog.parts = fitted[n][0]
                with torch.no_grad():
                    q = prog(ie)
                    per[n].append(kl_t(pe, q).reshape(-1).numpy())
                    top1[n].append((pe.argmax(-1) == q.argmax(-1)).reshape(-1).numpy())
                    qarg[n].append(q.argmax(-1).reshape(-1).numpy())
                    if Ae is not None:
                        for r, h in (("h0", 0), ("h1", 1), ("h2", 2)):
                            if r in prog.parts or (r == "h1" and "recall" in prog.parts):
                                pk[n][r].append(vp.pattern_kl(Ae[:, h], prog.attn(r, vp.sub(ie, slice(0, 4)))))
            log(f"{split} {c + len(w)}/{len(wins)}")
        res[split] = {}
        for n in names:
            k = np.concatenate(per[n])
            pw = k.reshape(-1, T).mean(1)
            res[split][n] = {"kl": float(k.mean()), "se": float(pw.std(ddof=1) / np.sqrt(len(pw))),
                             "top1": float(np.concatenate(top1[n]).mean()),
                             "pattern_kl": {r: float(np.mean(v)) for r, v in pk[n].items() if v}}
            log(f"{split} {n:50s} KL {k.mean():.4f} +- {res[split][n]['se']:.4f} top1 {res[split][n]['top1']:.3f} "
                f"pattern KL {res[split][n]['pattern_kl']}")
        if split == "train_eval":
            kfin = np.concatenate(per["L4 + PHRASE (trigram, 4-gram)"])
            k1 = np.concatenate(per["L1 FOLLOW"])
            conf = np.concatenate(plist[0::2])
            am = np.concatenate(plist[1::2])
            np.savez_compressed(f"{HERE}/v1ledger_perpos.npz", final=kfin, follow=k1, conf=conf, argmax=am)
            res["residual"] = residual_slices(wins, kfin, k1, conf, am, TOKENS)
            xf = np.asarray(wins)[:, :T].reshape(-1)
            pos = np.tile(np.arange(T), len(wins))
            echo = {"model top-1 = current token, position 0": float((am[pos == 0] == xf[pos == 0]).mean()),
                    "model top-1 = current token, positions >= 1": float((am[pos > 0] == xf[pos > 0]).mean())}
            for n in names:
                qa = np.concatenate(qarg[n])
                echo[f"{n}: program top-1 = current token, position 0"] = float((qa[pos == 0] == xf[pos == 0]).mean())
                echo[f"{n}: program top-1 = current token, positions >= 1"] = float((qa[pos > 0] == xf[pos > 0]).mean())
            res["echo"] = echo
            log("echo:", echo)
    out["eval"] = res
    json.dump(out, open(f"{HERE}/v1ledger_results.json", "w"), indent=1)
    log("done")


def residual_slices(wins, k, k1, conf, am, TOKENS):
    """Where the final rung still differs from the model (held-out TRAIN positions)."""
    F2 = np.load(f"{vp.REPO}/analysis/scratch/h2card/features.npz")
    cont, piece = F2["cont"], F2["is_cont"].astype(bool)
    x = np.asarray(wins)[:, :T]
    xf = x.reshape(-1)
    tot = k.sum()
    out = {"mean_kl": float(k.mean()), "slices": {}}

    def put(name, m):
        out["slices"][name] = {"share_positions": float(m.mean()), "kl_per_pos": float(k[m].mean()),
                               "share_kl": float(k[m].sum() / tot), "follow_kl_per_pos": float(k1[m].mean())}

    onset = cont[xf] > 0.5
    put("current token is a word onset (P(next is a word piece) > 0.5)", onset)
    put("current token is a word piece", piece[xf])
    put("inside a word (onset or piece)", onset | piece[xf])
    pos = np.tile(np.arange(T), len(x))
    put("window position 0", pos == 0)
    ii = np.arange(T)
    start = np.array([np.maximum.accumulate(np.where(r == 0, ii, -1)) for r in x])
    since = np.where(start >= 0, ii[None, :] - start, 10 ** 6).reshape(-1)
    put("current token is end-of-story", xf == 0)
    put("1-4 tokens after a story start", (since >= 1) & (since <= 4))
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        put(f"model top-1 probability in [{lo}, {hi})", (conf >= lo) & (conf < hi))
    cls, names = class_index("CONTEXT")
    c = cls[xf]
    share = np.bincount(c, weights=k, minlength=len(names)) / tot
    occ = np.bincount(c, minlength=len(names))
    out["by_context_class_top12"] = [
        {"class": names[j], "share_kl": float(share[j]), "share_positions": float(occ[j] / len(k)),
         "kl_per_pos": float(share[j] * tot / max(occ[j], 1))} for j in np.argsort(-share)[:12]]
    worst = []
    for q in np.argsort(-k)[:15]:
        b, i = divmod(int(q), T)
        ctxs = "".join(TOKENS[t] for t in x[b, max(0, i - 12): i + 1]).replace("\n", "\\n")
        worst.append({"kl": float(k[q]), "context": ctxs[-60:], "model_top": TOKENS[int(am[q])],
                      "model_top_p": float(conf[q])})
    out["worst_positions"] = worst
    return out


if __name__ == "__main__":
    main()
