"""KL between a model and program v1.3 on the cached evaluation sets (analysis/readable_training/evalsets.py).
Tools b1 (dev_fast; the logged number comes from the harness, this reproduces it) and c (out of distribution).

    .venv/bin/python -m analysis.readable_training.fidelity <model id or checkpoint> [set ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from analysis.readable_training import evalsets as E

V = 2000


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def on_set(model, name, chunk=25) -> dict:
    """Per-window KL(model || program) and KL(program || model), top-1 agreement, and both cross-entropies on the
    windows' next tokens (meaningful for text sets)."""
    from harness.common import model_logprobs

    W = E.windows(name)
    prog = E.program_lp(name)
    kl_mp, kl_pm, top1, ce_m, ce_p = [], [], [], [], []
    for c in range(0, len(W), chunk):
        lp = model_logprobs(model, W[c:c + chunk])
        lq = lsm(np.asarray(prog[c:c + chunk], np.float64))
        p, q = np.exp(lp), np.exp(lq)
        kl_mp += list((p * (lp - lq)).sum(-1).mean(1))
        kl_pm += list((q * (lq - lp)).sum(-1).mean(1))
        top1 += list((lp.argmax(-1) == lq.argmax(-1)).mean(1))
        nxt = W[c:c + chunk, 1:257]
        ce_m += list(-np.take_along_axis(lp, nxt[..., None], -1)[..., 0].mean(1))
        ce_p += list(-np.take_along_axis(lq, nxt[..., None], -1)[..., 0].mean(1))
    a = lambda x: np.asarray(x, np.float64)   # noqa: E731
    return {"kl_model_prog": float(a(kl_mp).mean()), "kl_prog_model": float(a(kl_pm).mean()),
            "top1": float(a(top1).mean()), "ce_model": float(a(ce_m).mean()), "ce_program": float(a(ce_p).mean()),
            "per_window_kl_model_prog": [float(x) for x in kl_mp], "n_windows": len(W)}


def run(model_path, sets) -> dict:
    from harness.common import load_model

    m = load_model(model_path)
    return {name: on_set(m, name) for name in sets}


if __name__ == "__main__":
    sets = sys.argv[2:] or ["dev_fast"]
    res = run(sys.argv[1], sets)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_window")} for k, v in res.items()},
                     indent=1))
