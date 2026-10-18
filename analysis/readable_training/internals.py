"""Internal tools for the readable-training measurements, on a head-aligned checkpoint (models.py): heads 0, 1, 2 play
LOCAL-0, RECALL, LOCAL-2.

  d3  attention pattern KL between model head h and rule h, both directions, on train_windows(16, 9125) and on the
      first 16 windows of each out-of-distribution set
  d4  vote (OV) row correlation of heads 0-2 with the vote targets (data/readable/vote_targets.npz), row-weighted and
      uniform
  d5  each head's mean-ablation shift: KL(model || model with that head's output at its mean), train_windows(64, 9100)
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

from analysis.readable_training.common import REPO, RULE_HEADS

PATTERN = (16, 9125)


def pattern_kls(model, W, s) -> dict:
    """Mean over query rows i >= 1 of KL(model head || rule) and KL(rule || model head), for heads 0-2."""
    from analysis.attention_fit import model_attention

    A = model_attention(model, W).astype(np.float64)
    out = {}
    for i, r in enumerate(RULE_HEADS):
        mod = importlib.import_module(f"program.{r}")
        mr, rm, n = 0.0, 0.0, 0
        for w, a in zip(W, A[:, i]):
            R = mod.attention(np.asarray(w[:256], np.int64), s)
            a1, r1 = a[1:], R[1:]
            la, lr = np.log(np.maximum(a1, 1e-30)), np.log(np.maximum(r1, 1e-30))
            mr += (a1 * (la - lr)).sum()
            rm += (r1 * (lr - la)).sum()
            n += a1.shape[0]
        out[f"h{i}"] = {"kl_model_rule": mr / n, "kl_rule_model": rm / n}
    out["mean_kl_model_rule"] = float(np.mean([out[f"h{i}"]["kl_model_rule"] for i in range(3)]))
    out["mean_kl_rule_model"] = float(np.mean([out[f"h{i}"]["kl_rule_model"] for i in range(3)]))
    return out


def vote_rhos(model) -> dict:
    from src.train import readable as R

    tg = np.load(f"{REPO}/data/readable/vote_targets.npz")
    targets, colw, roww = (torch.as_tensor(tg[k]) for k in ("targets", "colw", "roww"))
    with torch.no_grad():
        ov = R.ov_tables_torch(model)
        out = {}
        for i in range(3):
            rho = R.row_corr(ov[i], targets[i], colw)
            out[f"h{i}"] = {"weighted": float((roww[i] * rho).sum()), "uniform": float(rho.mean())}
    out["mean_weighted"] = float(np.mean([out[f"h{i}"]["weighted"] for i in range(3)]))
    out["mean_uniform"] = float(np.mean([out[f"h{i}"]["uniform"] for i in range(3)]))
    return out


def ablation_shifts(model, W) -> dict:
    from harness.common import head_means, model_logprobs

    means = head_means(model, W)
    lp = model_logprobs(model, W)
    out = {}
    for h in range(4):
        la = model_logprobs(model, W, ablate_heads=[h], ablation="mean", means=means)
        out[f"h{h}"] = float((np.exp(lp) * (lp - la)).sum(-1).mean())
    return out


def run(model_path, ood_sets) -> dict:
    from analysis.fitlib import train_windows
    from analysis.mech_eval import stats
    from analysis.readable_training import evalsets as E
    from harness.common import load_model

    s = stats()
    m = load_model(model_path)
    res = {"d3_pattern": pattern_kls(m, train_windows(*PATTERN), s),
           "d3_pattern_ood": {name: pattern_kls(m, E.windows(name)[:16], s) for name in ood_sets},
           "d4_vote": vote_rhos(m),
           "d5_ablation": ablation_shifts(m, train_windows(64, 9100))}
    return res
