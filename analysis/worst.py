"""Where does the program lose KL? Groups per-position KL(model || program) on dev_fast.

  python -m analysis.worst --model attn_s0 [--program program] [--n 50]

Prints KL share by class of the current token, by class of the model's top token, by position bucket,
and the worst individual positions with context. Runs the program in-process (open-track tool).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.common import eval_windows, load_model, model_logprobs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--program", default="program")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    parent, name = os.path.split(os.path.abspath(args.program))
    sys.path.insert(0, parent)
    prog = importlib.import_module(name)
    from program.categories import CLASS_OF
    from program.vocab import TOKENS

    W = eval_windows("dev_fast")[: args.n]
    lp = model_logprobs(load_model(args.model), W)
    rows = []
    for k, w in enumerate(W):
        x = w[:256].astype(int)
        q = prog.predict(x.tolist())
        p = np.exp(lp[k])
        kl = (p * (lp[k] - q)).sum(1)
        for i in range(256):
            rows.append((kl[i], k, i, x[i], int(lp[k][i].argmax()), int(q[i].argmax())))
    tot = sum(r[0] for r in rows)
    n = len(rows)
    print(f"mean KL {tot / n:.4f} over {n} positions")

    def report(title, keyf):
        agg = defaultdict(lambda: [0.0, 0])
        for r in rows:
            a = agg[keyf(r)]
            a[0] += r[0]
            a[1] += 1
        print(f"\n== {title}: share of total KL, count, mean KL")
        for key, (s, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:args.top]:
            print(f"  {str(key):32s} {s / tot:6.1%} {c:6d} {s / c:.3f}")

    report("class of current token", lambda r: CLASS_OF[r[3]])
    report("class of model's top token", lambda r: CLASS_OF[r[4]])
    report("position bucket", lambda r: min(r[2] // 32, 7) * 32)
    report("current token", lambda r: repr(TOKENS[r[3]]))
    print("\n== worst positions")
    for kl, k, i, cur, mt, pt in sorted(rows, reverse=True)[: args.top]:
        ctx = "".join(TOKENS[t] for t in W[k][max(0, i - 12) : i + 1])
        print(f"  KL {kl:.2f}  ...{ctx!r} -> model {TOKENS[mt]!r}, program {TOKENS[pt]!r}")


if __name__ == "__main__":
    main()
