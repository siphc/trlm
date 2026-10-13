"""Print examples at word-internal positions: model, model with heads 0+2(+3) earlier parts mean-replaced, program,
corpus word-prefix distribution. Also aggregate: how much of the heads' effect is a shift of mass between word
pieces and non-pieces (does the word go on?).

  python -m analysis.investigations.word.examples
"""
from __future__ import annotations

import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes, word_dist
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M
from program.vocab import TOKENS


def top(lp, k=6):
    idx = np.argsort(-lp)[:k]
    return "  ".join(f"{TOKENS[j]!r}:{np.exp(lp[j]):.2f}" for j in idx)


def main(n=16):
    from harness.common import head_means, load_model

    P = program()
    PIECE = WS.piece_flags()
    wk, wc, _, _ = WS.load()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    lpa = model_logprobs(W, ablate_heads=[0, 2], ablation="earlier", means=means)
    lpb = model_logprobs(W, ablate_heads=[0, 2, 3], ablation="earlier", means=means)
    lpc = model_logprobs(W, ablate_heads=[0, 1, 2, 3], ablation="earlier", means=means)
    agg = {}
    shown = 0
    for w, p, pa, pb, pc in zip(W, lp, lpa, lpb, lpc):
        t = np.asarray(w[:256], dtype=np.int64)
        q = P.predict(t)
        cl, Ls = classes(P, t, PIECE)
        for i in range(1, len(t)):
            c = cl[i]
            pm = np.exp(p[i])
            ppiece = lambda l: float(np.exp(l)[PIECE].sum())
            r = agg.setdefault(c, np.zeros(8))
            r += [1, ppiece(p[i]), ppiece(pa[i]), ppiece(pb[i]), ppiece(pc[i]), ppiece(q[i]),
                  M.kl(p[i], pa[i]), M.kl(p[i], pc[i])]
            if c.startswith("inside") and shown < 40 and i % 3 == 0:
                shown += 1
                ctx = "".join(TOKENS[x] for x in t[max(0, i - 8) : i + 1]).replace("\n", "\\n")
                print(f"--- ...{ctx!r}  [{c}]  next={TOKENS[t[i + 1]]!r}" if i + 1 < len(t) else "")
                print("  model    ", top(p[i]))
                print("  -h02     ", top(pa[i]), f"KL {M.kl(p[i], pa[i]):.2f}")
                print("  -h023    ", top(pb[i]), f"KL {M.kl(p[i], pb[i]):.2f}")
                print("  -all     ", top(pc[i]), f"KL {M.kl(p[i], pc[i]):.2f}")
                print("  program  ", top(q[i]), f"KL {M.kl(p[i], q[i]):.2f}")
                pw, _ = word_dist(P, t, i, 3.0, PIECE, wk, wc)
                print("  corpus Pw", top(np.log(pw)))
    print("\nclass: n, P(next piece) model / -h02 / -h023 / -all / program, KL(model||-h02), KL(model||-all)")
    for c, r in sorted(agg.items()):
        print(f"  {c:25s} {int(r[0]):5d}  " + "  ".join(f"{x / r[0]:.3f}" for x in r[1:]))


if __name__ == "__main__":
    main()
