"""Apply the pre-registered decision rules (logs/prereg_readable_training.md) and record their values in
analysis/out/readable_training/rules.json. Rule R1 is applied by build_pool.py coverage.

  .venv/bin/python -m analysis.readable_training.apply_rules r2      # c = L_att / L_vote on attn_s0 (held-out pool)
  .venv/bin/python -m analysis.readable_training.apply_rules r6      # per-step time of each arm on MPS
  .venv/bin/python -m analysis.readable_training.apply_rules r3      # lambda* per arm from the 100M-token grid
  .venv/bin/python -m analysis.readable_training.apply_rules r5      # loss spikes in the grid runs
  .venv/bin/python -m analysis.readable_training.apply_rules r7      # memorisation gap of the seed-0 one-epoch runs
  .venv/bin/python -m analysis.readable_training.apply_rules r3full  # R3's second part: seed-0 one-epoch cost
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


DEV = os.path.join(REPO, "models", "readable", "dev")
GRID = [0.1, 0.3, 1, 3, 10]


def _result(name):
    p = os.path.join(DEV, name, "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def cmd_r3():
    """R3: per arm, lambda* = the largest grid value whose 100M-token run has dev_full CE <= 1.01 x the lambda = 0 run's
    (0.1 if none)."""
    base = _result("none_s0_100M")["dev_full_loss"]
    res = {"lambda0_dev_full": base}
    for arm in ("O", "I"):
        rows = {lam: _result(f"{arm}_l{lam:g}_s0_100M") for lam in GRID}
        assert all(rows.values()), f"grid incomplete for arm {arm}"
        cost = {f"{lam:g}": r["dev_full_loss"] / base - 1 for lam, r in rows.items()}
        ok = [lam for lam in GRID if rows[lam]["dev_full_loss"] <= 1.01 * base]
        res[arm] = {"relative_cost": cost, "lambda_star": max(ok) if ok else 0.1}
    save_rules({"R3": res})
    print(json.dumps(res, indent=1))


def spikes(run_dir, warm_frac=0.01, window=200, thresh=0.1):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    rows = {r["step"]: r["train_loss"] for r in rows}
    total = max(rows)
    warm = int(round(warm_frac * total))
    out = []
    for s, v in sorted(rows.items()):
        if s <= warm:
            continue
        prev = [rows[t] for t in rows if s - window <= t < s and t > warm]
        if prev and v - min(prev) > thresh:
            out.append((s, round(v - min(prev), 3)))
    return out


def cmd_r5():
    """R5: a loss spike = a rise of more than 0.1 nats within 200 steps after warmup (logged training loss); if any run
    of an arm spikes, all runs of that arm ramp lambda over the first 5% of steps."""
    res = {}
    for name in sorted(os.listdir(DEV)):
        if _result(name):
            res[name] = spikes(os.path.join(DEV, name))
    ramp = {f"ramp_{arm}": 0.05 if any(v for n, v in res.items() if n.startswith(arm + "_")) else 0.0
            for arm in ("O", "I")}
    save_rules({"R5": {"spikes": res, **ramp}})
    print(json.dumps({"spikes": {k: v for k, v in res.items() if v}, **ramp}, indent=1))


def cmd_r3full():
    """R3, second part: if a seed-0 one-epoch run at lambda* costs more than 2% dev_full CE against attn_s0, step down
    one grid value and repeat the seed-0 run. Re-run after each repeated run: the check continues from the last
    lambda until one passes (or the lowest grid value is reached)."""
    rules = load_rules()
    base = json.load(open(os.path.join(REPO, "models", "final", "attn_s0", "result.json")))["dev_full_loss"]
    old = rules.get("R3full", {})
    res = {"attn_s0_dev_full": base}
    for arm in ("O", "I"):
        hist = old.get(arm, {}).get("history", [])
        lam = old.get(arm, {}).get("lambda_final", rules["R3"][arm]["lambda_star"])
        p = os.path.join(REPO, "models", "readable", f"{arm}_l{lam:g}_s0", "result.json")
        if not os.path.exists(p):
            res[arm] = {"history": hist, "lambda_final": lam, "status": f"needs the seed-0 run at lambda {lam:g}"}
            continue
        if not any(h["lambda"] == lam for h in hist):
            cost = json.load(open(p))["dev_full_loss"] / base - 1
            hist.append({"lambda": lam, "relative_cost": cost})
        cost = next(h["relative_cost"] for h in hist if h["lambda"] == lam)
        if cost > 0.02 and GRID.index(lam) > 0:
            res[arm] = {"history": hist, "lambda_final": GRID[GRID.index(lam) - 1],
                        "status": f"stepped down from {lam:g}; needs the seed-0 run at the new lambda"}
        else:
            res[arm] = {"history": hist, "lambda_final": lam, "status": "final"}
    save_rules({"R3full": res})
    print(json.dumps(res, indent=1))


def cmd_r7():
    """R7: if a seed-0 one-epoch run's memorisation gap (held-out minus pool-training trained loss, last evaluation)
    exceeds 10% of its held-out value, the pool grows to 65,536 windows before confirmatory runs."""
    rules = load_rules()
    res = {}
    for arm, key in (("O", "out"), ("I", "att")):
        lam = rules.get("R3full", {}).get(arm, {}).get("lambda_final") or rules["R3"][arm]["lambda_star"]
        d = os.path.join(REPO, "models", "readable", f"{arm}_l{lam:g}_s0")
        rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if "m_heldout_out_coarse" in l]
        last = rows[-1]
        ho, tr = last[f"m_heldout_{key}_coarse"], last[f"m_train_{key}_coarse"]
        res[arm] = {"heldout": ho, "train": tr, "gap": ho - tr, "relative_gap": (ho - tr) / ho, "grow": (ho - tr) / ho > 0.10}
    res["grow_pool"] = any(v["grow"] for v in res.values())
    save_rules({"R7": res})
    print(json.dumps(res, indent=1))


def tool_values(suite: dict, seed: int) -> dict:
    """The scalar value of every pre-registered tool in one suite.json (lower is better for KLs, higher for rho)."""
    f, i = suite["fidelity"], suite["internals"]
    v = {"b1_harness_kl": suite["harness"]["kl"] if "harness" in suite and "kl" in suite["harness"]
         else f["dev_fast"]["kl_model_prog"],
         "b1_cached_kl": f["dev_fast"]["kl_model_prog"],
         "train_eval_kl": f["train_eval"]["kl_model_prog"],
         "d3_pattern_kl": i["d3_pattern"]["mean_kl_model_rule"],
         "d4_vote_rho_uniform": i["d4_vote"]["mean_uniform"], "d4_vote_rho_weighted": i["d4_vote"]["mean_weighted"]}
    for name in [f"probe_attn_s{seed}", "program_samples", "sent_shuffle", "word_shuffle"]:
        key = "probe_twin" if name.startswith("probe") else name
        v[f"c_kl_{key}"] = f[name]["kl_model_prog"]
        v[f"d3_pattern_kl_{key}"] = i["d3_pattern_ood"][name]["mean_kl_model_rule"]
    v["c_kl_ood_mean"] = float(np.mean([v[f"c_kl_{k}"] for k in ("probe_twin", "program_samples", "sent_shuffle",
                                                                    "word_shuffle")]))
    v["d3_pattern_kl_ood_mean"] = float(np.mean([v[f"d3_pattern_kl_{k}"] for k in ("probe_twin", "program_samples",
                                                                                   "sent_shuffle", "word_shuffle")]))
    for h in range(4):
        v[f"d5_ablation_h{h}"] = i["d5_ablation"][f"h{h}"]
    if "lock" in suite:
        v["d2_lock"] = suite["lock"]["L"]
    if "refit" in suite and "kl" in suite["refit"]["harness"]:
        v["b2_refit_kl"] = suite["refit"]["harness"]["kl"]
    return v


def cmd_r4():
    """R4: sigma_T = the standard deviation of each tool over the aligned attn_s1..s4 (suite.json)."""
    from analysis.readable_training.common import OUT

    vals = {}
    for k in range(1, 5):
        su = json.load(open(os.path.join(OUT, f"attn_s{k}", "suite.json")))
        vals[f"attn_s{k}"] = tool_values(su, k)
    keys = sorted(set.intersection(*(set(v) for v in vals.values())))
    res = {"values": vals, "sigma": {t: float(np.std([vals[m][t] for m in vals], ddof=1)) for t in keys}}
    save_rules({"R4": res})
    print(json.dumps(res["sigma"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=["r2", "r6", "r3", "r5", "r7", "r3full", "r4"])
    a = ap.parse_args()
    {"r2": cmd_r2, "r6": cmd_r6, "r3": cmd_r3, "r5": cmd_r5, "r7": cmd_r7, "r3full": cmd_r3full,
     "r4": cmd_r4}[a.rule]()


if __name__ == "__main__":
    main()
