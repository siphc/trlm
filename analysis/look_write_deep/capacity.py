"""Capacity comparison (exploratory; logs/deep_study.md): do heads share and cancel less work when a model has more of
them? Per model, on train_windows(64, 2025):

- single-head earlier-token ablation (common_deep.forward ablate_ctx: z_h(i) = a_ii v_i + mean earlier part):
  D_h = lp - lp_abl(h), KL_h = KL(lp || lp_abl(h));
- rho(h, h'): pooled probability-weighted correlation of D_h and D_h' over the vocabulary (each centred per position
  under the model's distribution), pooled over all positions;
- cancellation share: head pairs with rho < -0.1;
- non-additivity of joint ablations: KL(joint h, h') / (KL_h + KL_h').
Stated in advance: the essay ("Neural Networks Are Already Clean") predicts broader, more shared work with fewer heads,
so the cancellation share and the mean negative rho should fall from 4 heads (1 block) to 8 (2 blocks) to 16 (4 blocks).

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.capacity attn_s0 attn2L_s0 ...
Writes analysis/out/look_write_deep/<model>/capacity.json.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from analysis.look_write_deep import common_deep as C

CHUNK = 8


def run(model_id):
    torch.set_num_threads(2)
    t0 = time.time()
    m = C.load(model_id)
    L, H, _, _ = C.dims(m)
    heads = [(l, h) for l in range(L) for h in range(H)]
    W = C.windows(C.EVAL_SEED, C.N_EVAL)
    means = C.ctx_means(m, W)
    nH = len(heads)
    gram = np.zeros((nH, nH))
    kl_single = np.zeros(nH)
    pairs = list(itertools.combinations(range(nH), 2))
    kl_joint = np.zeros(len(pairs))
    n_pos = 0
    with torch.no_grad():
        for s in range(0, len(W), CHUNK):
            x = torch.as_tensor(np.asarray(W[s:s + CHUNK, :C.T], np.int64))
            lp = torch.log_softmax(C.forward(m, x)[0].double(), -1)
            p = lp.exp()
            deltas = []
            for i, (l, h) in enumerate(heads):
                lpa = torch.log_softmax(C.forward(m, x, ablate_ctx={(l, h): means[(l, h)]})[0].double(), -1)
                kl_single[i] += float((p * (lp - lpa)).sum())
                d = lp - lpa
                d = d - (p * d).sum(-1, keepdim=True)                  # centred per position under the model
                deltas.append(d)
            for i in range(nH):
                for j in range(i, nH):
                    gram[i, j] += float((p * deltas[i] * deltas[j]).sum())
            for k, (i, j) in enumerate(pairs):
                abl = {heads[i]: means[heads[i]], heads[j]: means[heads[j]]}
                lpa = torch.log_softmax(C.forward(m, x, ablate_ctx=abl)[0].double(), -1)
                kl_joint[k] += float((p * (lp - lpa)).sum())
            n_pos += x.numel()
    gram = np.triu(gram) + np.triu(gram, 1).T
    rho = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    kl_single /= n_pos
    kl_joint /= n_pos
    pair_rows = []
    for k, (i, j) in enumerate(pairs):
        pair_rows.append({"a": f"L{heads[i][0]}H{heads[i][1]}", "b": f"L{heads[j][0]}H{heads[j][1]}",
                          "rho": float(rho[i, j]), "nonadditivity": float(kl_joint[k] / (kl_single[i] + kl_single[j])),
                          "same_layer": heads[i][0] == heads[j][0]})
    rhos = np.array([r["rho"] for r in pair_rows])
    res = {"model": model_id, "n_blocks": L, "n_heads": nH,
           "kl_single": {f"L{l}H{h}": float(kl_single[i]) for i, (l, h) in enumerate(heads)},
           "pairs": pair_rows,
           "summary": {"cancellation_share": float((rhos < -0.1).mean()), "mean_rho": float(rhos.mean()),
                       "mean_negative_rho": float(rhos[rhos < 0].mean()) if (rhos < 0).any() else 0.0,
                       "mean_abs_rho": float(np.abs(rhos).mean()),
                       "mean_nonadditivity": float(np.mean([r["nonadditivity"] for r in pair_rows]))},
           "seconds": time.time() - t0}
    with open(os.path.join(C.model_dir(model_id), "capacity.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"{model_id}: {nH} heads, " + ", ".join(f"{k} {v:.3f}" for k, v in res["summary"].items())
          + f" [{res['seconds']:.0f}s]", flush=True)
    return res


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        run(mid)
