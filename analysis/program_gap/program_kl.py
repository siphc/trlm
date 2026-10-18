"""Per-window KL(model || program) for a program directory, on TRAIN windows (for paired comparisons of two
program versions, one process each so the two `program` packages never meet).

    .venv/bin/python analysis/program_gap/program_kl.py --parent /path/containing/program --seed 2000 --n 64 --out f.npy
    .venv/bin/python analysis/program_gap/program_kl.py --compare a.npy b.npy
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent")
    ap.add_argument("--name", default="program")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (np.load(f) for f in a.compare)
        d = y - x
        print(f"{a.compare[0]}: {x.mean():.5f}  {a.compare[1]}: {y.mean():.5f}  diff {d.mean():+.5f} +- "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.5f}  (n={len(d)})")
        return
    sys.path.insert(0, a.parent)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(1, repo)
    from analysis.fitlib import train_windows
    from analysis.mech_eval import stats
    from harness.common import load_model, model_logprobs

    import importlib
    P = importlib.import_module(a.name)
    assert os.path.dirname(os.path.dirname(P.__file__)) == os.path.abspath(a.parent), P.__file__
    P.STATS = stats()
    W = train_windows(a.n, seed=a.seed)
    lp = model_logprobs(load_model(a.model), W)
    out = []
    for w, p in zip(W, lp):
        q = P.predict(np.asarray(w[:256], np.int64))
        out.append(float((np.exp(p) * (p - q)).sum(-1).mean()))
    np.save(a.out, np.asarray(out))
    print(a.parent, np.mean(out))


if __name__ == "__main__":
    main()
