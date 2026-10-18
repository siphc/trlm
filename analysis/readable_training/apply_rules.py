"""Apply the pre-registered decision rules (logs/prereg_readable_training.md) and record their values in
analysis/out/readable_training/rules.json. Rule R1 is applied by build_pool.py coverage.

  .venv/bin/python -m analysis.readable_training.apply_rules r2      # c = L_att / L_vote on attn_s0 (held-out pool)
  .venv/bin/python -m analysis.readable_training.apply_rules r6      # per-step time of each arm on MPS
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from analysis.readable_training.common import REPO, load_rules, pool_dir, save_rules
from src.train import readable as R


def cmd_r2():
    """R2: c such that L_att and c * L_vote are equal on attn_s0 (its heads 0, 1, 2 play LOCAL-0, RECALL, LOCAL-2),
    on the 512 held-out pool windows, with the trained (coarsened) attention loss."""
    from src.train.train import load_model

    rules = load_rules()
    ks = [rules["R1"]["k_att"][h] for h in ("head0", "head1", "head2")]
    pool = R.Pool(pool_dir())
    m = load_model(os.path.join(REPO, "models", "final", "attn_s0", "final.pt"))
    tg = np.load(os.path.join(REPO, "data", "readable", "vote_targets.npz"))
    att = []
    with torch.no_grad():
        for c in range(0, pool.n_heldout, 32):
            r = np.arange(c, min(c + 32, pool.n_heldout))
            x = torch.as_tensor(np.asarray(pool.tokens[pool.meta["n_train"] + r, :R.T], np.int64))
            rule = torch.as_tensor(np.asarray(pool.heldout_att[r])).float()
            rule = rule / rule.sum(-1, keepdim=True).clamp_min(1e-30)
            _, pats = m(x, return_patterns=True)
            per = []
            for i, h in enumerate(R.HEADS):
                pv, pk = rule[:, i].topk(ks[i], -1)
                per.append(float(R.attention_topk_kl(pats[0][:, h], pk, pv, (1 - pv.sum(-1)).clamp_min(0))))
            att.append(np.mean(per))
        ov = R.ov_tables_torch(m)
        targets = torch.as_tensor(tg["targets"])
        colw, roww = torch.as_tensor(tg["colw"]), torch.as_tensor(tg["roww"])
        rho = torch.stack([R.row_corr(ov[i], targets[i], colw) for i in range(3)])
        l_vote = float((roww * (1 - rho)).sum(1).mean())
    l_att = float(np.mean(att))
    res = {"L_att_attn_s0": l_att, "L_vote_attn_s0": l_vote, "c": l_att / l_vote,
           "vote_rho_weighted_attn_s0": [float((roww[i] * rho[i]).sum()) for i in range(3)],
           "vote_rho_uniform_attn_s0": [float(rho[i].mean()) for i in range(3)]}
    save_rules({"R2": res})
    print(json.dumps(res, indent=1))


def cmd_r6(steps=150):
    """R6: the per-step time of a baseline step and of each arm's step (lambda 1), batch 64 x 256, 16 auxiliary
    windows, on MPS; 8 auxiliary windows if an arm's step takes more than 1.6 x the baseline's."""
    from src.train.train import TrainConfig, train

    tmp = tempfile.mkdtemp(prefix="readable_r6_")
    per = {}
    try:
        for name, kw in (("baseline", {}), ("O", dict(readable_arm="O", lambda_out=1.0)),
                         ("I", dict(readable_arm="I", lambda_att=1.0, lambda_vote=1.0))):
            out = os.path.join(tmp, name)
            train(TrainConfig(variant="attn", seed=0, budget_tokens=steps * 64 * 256, peak_lr=2e-3, device="mps",
                              eval_every=10**9, log_every=50, out=out, **kw))
            rows = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
            e = {r["step"]: r["elapsed"] for r in rows}
            per[name] = (e[steps] - e[50]) / (steps - 50)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ratio = {k: per[k] / per["baseline"] for k in ("O", "I")}
    res = {"seconds_per_step": per, "ratio_to_baseline": ratio,
           "aux_windows": 8 if max(ratio.values()) > 1.6 else 16}
    save_rules({"R6": res})
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=["r2", "r6"])
    a = ap.parse_args()
    {"r2": cmd_r2, "r6": cmd_r6}[a.rule]()


if __name__ == "__main__":
    main()
