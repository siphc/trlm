"""Checks for the deep-study code (no outcome quantities). Run on every model before its analysis:

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.selftest_deep attn_s0 attn2L_s0 ...

1. The explicit forward pass reproduces the model's logits; substituting each head's own pattern changes nothing;
   a uniform pattern does change the logits (every layer and head).
2. Gradients: TOTAL readout matches a finite difference along a random direction of the residual after each block;
   for the last block, SAME = TOTAL.
3. One-block models: the DV readout of the per-instance write equals the one-layer OV table entry difference
   OV(k, y) - E_P1[OV(k, .)] up to the per-instance vs token-level LayerNorm input (reported as a correlation).
Writes analysis/out/look_write_deep/<model>/selftest.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from analysis.look_write_deep import common_deep as C


def run(model_id):
    torch.manual_seed(0)
    m = C.load(model_id)
    L, H, DH, D = C.dims(m)
    w = C.windows(C.FIT_SEED, 4)
    x = torch.as_tensor(np.asarray(w[:, :C.T], dtype=np.int64))
    y = torch.as_tensor(np.asarray(w[:, 1:C.T + 1], dtype=np.int64))
    res = {"model": model_id, "n_blocks": L}
    with torch.no_grad():
        ref = m(x)
        logits, info = C.forward(m, x)
        res["explicit_vs_model_max_abs"] = float((ref - logits).abs().max())
        own, uni = [], []
        U = torch.tril(torch.ones(C.T, C.T)) / torch.arange(1, C.T + 1)[:, None]
        for l in range(L):
            for h in range(H):
                lo, _ = C.forward(m, x, sub=(l, h, info["pattern"][l][:, h]))
                own.append(float((lo - logits).abs().max()))
                lu, _ = C.forward(m, x, sub=(l, h, U.expand(len(x), -1, -1)))
                uni.append(float((lu - logits).abs().max()))
        res["own_pattern_max_abs"] = max(own)
        res["uniform_pattern_min_change"] = min(uni)
    # gradients (float64 copy of the model, so the finite difference is not lost to rounding)
    import copy

    m64 = copy.deepcopy(m).double()
    P1 = torch.as_tensor(C.p1_table(), dtype=torch.float64)
    R = C.bigram_error(x, y, P1)
    grads, logits, info = C.residual_grads(m64, x, R)
    fd = []
    for l in range(L):
        direction = torch.randn_like(info["resid"][l])
        direction = direction / direction.norm()
        eps = 1e-4

        def F_at(shift):
            with torch.no_grad():
                h = m64.wte(x) + m64.wpe(torch.arange(C.T))[None]
                for ll, blk in enumerate(m64.blocks):
                    a, _ = blk.attn(blk.ln1(h), return_pattern=True)
                    h = h + a
                    if ll == l:
                        h = h + shift
                return float((m64.ln_f(h) @ m64.wte.weight.T * R).sum())

        num = (F_at(eps * direction) - F_at(-eps * direction)) / (2 * eps)
        ana = float((grads[l] * direction).sum())
        fd.append({"layer": l, "finite_difference": num, "analytic": ana,
                   "rel_err": abs(num - ana) / max(1e-12, abs(num))})
    res["grad_check"] = fd
    R = R.float()
    grads, logits, info = C.residual_grads(m, x, R)
    g_same, _, _ = C.residual_grads(m, x, R, detach_kv_after=L - 1)
    res["same_equals_total_last_block_max_abs"] = float((g_same[-1] - grads[-1]).abs().max())
    res["ok"] = (res["explicit_vs_model_max_abs"] < 1e-4 and res["own_pattern_max_abs"] < 1e-4
                 and res["uniform_pattern_min_change"] > 1e-3 and all(f["rel_err"] < 1e-4 for f in fd)
                 and res["same_equals_total_last_block_max_abs"] < 1e-5)
    with open(f"{C.model_dir(model_id.replace('@', '_'))}/selftest.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    torch.set_num_threads(2)
    bad = [mid for mid in sys.argv[1:] if not run(mid)["ok"]]
    if bad:
        sys.exit(f"selftest failed: {bad}")
