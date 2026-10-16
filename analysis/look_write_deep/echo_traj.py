"""Addendum D (logs/prereg_look_follows_write_addendum.md): does the look-follows-write relation appear before the
model is calibrated? One-layer models attn_s1..s4, checkpoints CKPTS.

Per checkpoint c, with that checkpoint's own weights:
  r(c)  pure-pair correlation at d = 1 between the head's exact query-key pair interaction (look_write
        common.pure_pair_scores) and its own usefulness table (BGU with the DV readout, = UOV, from a corpus sample)
  m(c)  correlation between the usefulness table and its model-echo version (calibration index)
Cross-lagged, pooled over the confirmatory LOCAL heads and consecutive checkpoints:
  corr(score_pp(c+1) - score_pp(c), U_pp(c))  vs  corr(U_pp(c+1) - U_pp(c), score_pp(c))
(pp = double-centred over the pairs with sample N >= 20 at d = 1, weights N.)

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.echo_traj [--tokens 5000000] [--device cpu]
Runs the sample pass for each checkpoint (pass_bgu) if its bgu.npz is missing, then writes
analysis/out/look_write_deep/echo_traj.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.look_write import common as C1
from analysis.look_write.fit_arms import double_centre, wcorr
from analysis.look_write_deep import common_deep as C
from analysis.look_write_deep import pass_bgu

MODELS = ["attn_s1", "attn_s2", "attn_s3", "attn_s4"]
CKPTS = ["ckpt_step229", "ckpt_step718", "ckpt_step2246", "ckpt_step7032", "ckpt_step22009", "final"]  # addendum D


def checkpoint_meta(model, ckpt):
    """Step, learning rate relative to peak, dev_fast loss nearest the step, and the final dev_fast loss."""
    rows = [json.loads(l) for l in open(os.path.join(C.REPO, "models", "final", model, "log.jsonl"))]
    evals = [r for r in rows if "dev_fast_loss" in r]
    peak = max(r["lr"] for r in rows)
    step = rows[-1]["step"] if ckpt == "final" else int(ckpt.replace("ckpt_step", ""))
    near = min(evals, key=lambda r: abs(r["step"] - step))
    lr_row = min(rows, key=lambda r: abs(r["step"] - step))
    return {"step": step, "lr_frac": lr_row["lr"] / peak, "dev_fast": near["dev_fast_loss"], "dev_fast_step": near["step"],
            "final_dev_fast": evals[-1]["dev_fast_loss"]}


def run(tokens, device):
    out = {"tokens": tokens, "models": {}}
    for model in MODELS:
        classes = json.load(open(os.path.join(C.model_dir(model), "classes.json")))["heads"]
        local = [c["head"] for c in classes if c["confirmatory"]]
        per = {}
        for ck in CKPTS:
            mid = f"{model}@{ck}"
            path = os.path.join(C.model_dir(mid), "bgu.npz")
            if not os.path.exists(path):
                pass_bgu.run(mid, tokens, device, 32)
            z = np.load(path)
            keys, N = z["keys"], z["N"]
            sel = (keys // (C.V * C.V) == 1) & (N >= 20)
            kk, qq, ww = (keys[sel] // C.V) % C.V, keys[sel] % C.V, N[sel]
            m = C1.load(C.model_path(mid))
            pps = C1.pure_pair_scores(C1.weights(m))                     # (H, V, V) [k, q]

            def dc(vals):
                Y = np.zeros((C.V, C.V))
                Y[kk, qq] = vals
                return double_centre(Y, kk, qq, ww, iters=1000, tol=1e-9)[0]

            heads = {}
            for h in range(len(pps)):
                U = dc(z["dv"][0, h][sel] / (N[sel] + C.SHRINK))
                E = dc(z["lecho"][0, h][sel] / (N[sel] + C.SHRINK))
                S = dc(pps[h][kk, qq])
                heads[h] = {"r": wcorr(S, U, ww), "m": wcorr(U, E, ww), "U": U, "S": S}
            per[ck] = {"meta": checkpoint_meta(model, ck), "heads": heads, "w": ww, "keys": keys[sel]}
            np.savez(os.path.join(C.model_dir(mid), "traj_tables.npz"), keys=keys[sel], w=ww,
                     U=np.stack([heads[h]["U"] for h in sorted(heads)]), S=np.stack([heads[h]["S"] for h in sorted(heads)]),
                     E=np.stack([dc(z["lecho"][0, h][sel] / (N[sel] + C.SHRINK)) for h in sorted(heads)]))
        out["models"][model] = {"local_heads": local, "per": per}
    # summary
    summary = {"by_checkpoint": [], "cross_lagged": {}}
    for ck_i, ck in enumerate(CKPTS):
        rows = []
        for model, mm in out["models"].items():
            for h in mm["local_heads"]:
                hd = mm["per"][ck]["heads"][h]
                rows.append({"model": model, "head": h, "r": hd["r"], "m": hd["m"], **mm["per"][ck]["meta"]})
        summary["by_checkpoint"].append({"checkpoint": ck, "heads": rows})
    num_a, num_b = [], []
    for model, mm in out["models"].items():
        for h in mm["local_heads"]:
            for a, b in zip(CKPTS[:-1], CKPTS[1:]):
                pa, pb = mm["per"][a], mm["per"][b]
                assert np.array_equal(pa["keys"], pb["keys"])
                w = pa["w"]
                num_a.append(wcorr(pb["heads"][h]["S"] - pa["heads"][h]["S"], pa["heads"][h]["U"], w))
                num_b.append(wcorr(pb["heads"][h]["U"] - pa["heads"][h]["U"], pa["heads"][h]["S"], w))
    summary["cross_lagged"] = {"corr(dScore, U_before) mean": float(np.mean(num_a)),
                               "corr(dU, Score_before) mean": float(np.mean(num_b)),
                               "pairs": len(num_a), "per_pair": list(zip(map(float, num_a), map(float, num_b)))}
    with open(os.path.join(C.OUT, "echo_traj.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for row in summary["by_checkpoint"]:
        rs = [x["r"] for x in row["heads"]]
        ms = [x["m"] for x in row["heads"]]
        meta = row["heads"][0]
        print(f"{row['checkpoint']}: step {meta['step']} lr {meta['lr_frac']:.2f} dev {meta['dev_fast']:.3f} "
              f"(final {meta['final_dev_fast']:.3f}) | r median {np.median(rs):.3f} min {min(rs):.3f} | "
              f"m median {np.median(ms):.3f}", flush=True)
    print(json.dumps({k: v for k, v in summary["cross_lagged"].items() if k != "per_pair"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(2)
    run(a.tokens, a.device)
