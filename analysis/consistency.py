"""Component consistency check (AGENTS.md section 9), with MEAN ablation on both sides.

For each head h:
  model side:    KL(model || model with head h mean-ablated)
                 (head h's per-position output z_h, before the output projection, replaced by its mean over
                  the reference windows; zero ablation is meaningless here, see logs/surprises.md)
                 and the earlier-token variant: keep a_ii v_i, mean-ablate sum_{j<i} a_ij v_j (matches programs
                 whose heads are silent on the current token)
  program side:  KL(program || program with head h's votes mean-ablated)      (self-effect, same definition)
                 KL(model || program without head h) - KL(model || program)  (fidelity change)
Disagreement between the two self-effects points at a component that does not do what its claim says.

  python -m analysis.consistency --model attn_s0 [--n 50]

The program is imported in-process (open-track tool). Writes analysis/out/<model>/consistency_<sha>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.common import eval_windows, load_model, program_sha  # noqa: E402


def kl_rows(p_log, q_log):
    p = np.exp(p_log)
    return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)


@torch.no_grad()
def model_logprobs_mean_ablated(model, W, heads=(), mean_z=None, bs=25):
    """Log-probs with the listed heads' outputs (before W_O) replaced by mean_z[h]."""
    H, Dh = model.cfg.n_heads, model.cfg.d_head
    handle = None
    if heads:
        def pre_hook(mod, args):
            z = args[0].clone()
            B, T, D = z.shape
            z = z.view(B, T, H, Dh)
            for h in heads:
                z[:, :, h, :] = mean_z[h]
            return (z.view(B, T, D),)
        handle = model.blocks[0].attn.out.register_forward_pre_hook(pre_hook)
    out = []
    try:
        for i in range(0, len(W), bs):
            x = torch.as_tensor(np.asarray(W[i:i + bs])[:, :256].astype(np.int64))
            out.append(torch.log_softmax(model(x).float(), -1).double().numpy())
    finally:
        if handle:
            handle.remove()
    return np.concatenate(out)


@torch.no_grad()
def head_mean_outputs(model, W, bs=25):
    """Mean of each head's output z_h (before W_O) over all positions of W: (H, Dh)."""
    H, Dh = model.cfg.n_heads, model.cfg.d_head
    acc, n = torch.zeros(H, Dh, dtype=torch.float64), 0
    store = {}

    def pre_hook(mod, args):
        store["z"] = args[0]
    handle = model.blocks[0].attn.out.register_forward_pre_hook(pre_hook)
    try:
        for i in range(0, len(W), bs):
            x = torch.as_tensor(np.asarray(W[i:i + bs])[:, :256].astype(np.int64))
            model(x)
            z = store["z"].double()
            acc += z.view(z.shape[0], z.shape[1], H, Dh).sum((0, 1))
            n += z.shape[0] * z.shape[1]
    finally:
        handle.remove()
    return (acc / n).float()


def model_context_ablation(M, W, W_ref, n_heads=4):
    """Earlier-token ablation in the exact numpy decomposition (analysis.tools.Model): for head h, keep the
    current-token part a_ii v_i of z_h(i) and replace the earlier-token part sum_{j<i} a_ij v_j by its mean
    over W_ref. Matches a program whose heads are silent on the current token. Returns
    (full log-probs, {h: ablated log-probs}) for W, each (n, 256, V)."""
    D = M.cfg.d_model
    Dh = D // n_heads

    def parts(tokens):
        t = np.asarray(tokens, dtype=int)
        n = len(t)
        x0 = M.WE[t] + M.WP[:n]
        h1, _ = M.ln(x0, M.g1)
        qkv = h1 @ M.Wqkv.T
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        mask = np.tril(np.ones((n, n), dtype=bool))
        selfp, ctx = [], []
        for h in range(n_heads):
            sl = slice(h * Dh, (h + 1) * Dh)
            sc = q[:, sl] @ k[:, sl].T / np.sqrt(Dh)
            sc = np.where(mask, sc, -np.inf)
            a = np.exp(sc - sc.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            d = np.diag(a)[:, None]
            selfp.append(d * v[:, sl])
            ctx.append(a @ v[:, sl] - d * v[:, sl])
        return x0, selfp, ctx

    def logprobs(x0, selfp, ctx):
        z = np.concatenate([selfp[h] + ctx[h] for h in range(n_heads)], axis=1)
        x = x0 + z @ M.WO.T
        hf, _ = M.ln(x, M.gf)
        lg = hf @ M.WE.T
        lg = lg - lg.max(1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(1, keepdims=True))

    mean_ctx = [np.zeros(Dh) for _ in range(n_heads)]
    cnt = 0
    for w in W_ref:
        _, _, ctx = parts(w[:256])
        for h in range(n_heads):
            mean_ctx[h] += ctx[h].sum(0)
        cnt += 256
    mean_ctx = [m / cnt for m in mean_ctx]
    full, abl = [], {h: [] for h in range(n_heads)}
    for w in W:
        x0, selfp, ctx = parts(w[:256])
        full.append(logprobs(x0, selfp, ctx))
        for h in range(n_heads):
            c2 = list(ctx)
            c2[h] = np.broadcast_to(mean_ctx[h], ctx[h].shape)
            abl[h].append(logprobs(x0, selfp, c2))
    return np.stack(full), {h: np.stack(v) for h, v in abl.items()}


def program_parts(prog, tokens):
    """FOLLOW votes and each head's votes for one window."""
    follow = prog.follow.votes(tokens)
    heads = [h.votes(tokens) for h in prog.HEADS]
    return follow, heads


def log_softmax(s):
    s = s - s.max(-1, keepdims=True)
    return s - np.log(np.exp(s).sum(-1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--program", default="program")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    parent, name = os.path.split(os.path.abspath(args.program))
    sys.path.insert(0, parent)
    import importlib

    prog = importlib.import_module(name)
    model = load_model(args.model)
    W = eval_windows("dev_fast")[: args.n]
    W_ref = eval_windows("claims")  # reference windows for the means (held out from dev_fast)
    mean_z = head_mean_outputs(model, W_ref)
    lp_model = model_logprobs_mean_ablated(model, W)
    H = len(prog.HEADS)

    # Program: full, and each head replaced by its mean vote vector over the reference windows.
    mean_votes = [np.zeros(2000) for _ in range(H)]
    count = 0
    for w in W_ref:
        _, hv = program_parts(prog, w[:256].astype(int))
        for h in range(H):
            mean_votes[h] += hv[h].sum(0)
        count += 256
    mean_votes = [m / count for m in mean_votes]

    res = {"model": args.model, "program_sha": program_sha(args.program), "n_windows": args.n, "heads": {}}
    from analysis.tools import Model

    _, ctx_abl = model_context_ablation(Model(args.model), W, W_ref[:50], n_heads=model.cfg.n_heads)
    kl_full_prog = []
    per_head = {h: {"model_self": [], "model_ctx": [], "prog_self": [], "prog_fid_removed": [], "prog_fid_mean": []}
                for h in range(H)}
    model_abl = {h: model_logprobs_mean_ablated(model, W, heads=(h,), mean_z=mean_z) for h in range(H)}
    for k, w in enumerate(W):
        t = w[:256].astype(int)
        follow, hv = program_parts(prog, t)
        full = follow + sum(hv)
        q = log_softmax(full)
        kl_full_prog.append(kl_rows(lp_model[k], q).mean())
        for h in range(H):
            q_mean = log_softmax(full - hv[h] + mean_votes[h][None, :])
            q_rm = log_softmax(full - hv[h])
            per_head[h]["model_self"].append(kl_rows(lp_model[k], model_abl[h][k]).mean())
            per_head[h]["model_ctx"].append(kl_rows(lp_model[k], ctx_abl[h][k]).mean())
            per_head[h]["prog_self"].append(kl_rows(q, q_mean).mean())
            per_head[h]["prog_fid_mean"].append(kl_rows(lp_model[k], q_mean).mean())
            per_head[h]["prog_fid_removed"].append(kl_rows(lp_model[k], q_rm).mean())
    base = float(np.mean(kl_full_prog))
    res["kl_model_program"] = base
    print(f"KL(model || program) on {args.n} dev_fast windows: {base:.4f}")
    print(f"{'head':>4} {'model: whole head':>18} {'model: earlier-token part':>26} {'program: KL(full||mean-abl)':>28} "
          f"{'fidelity loss (mean-abl)':>25} {'fidelity loss (removed)':>24}")
    for h in range(H):
        d = {k: float(np.mean(v)) for k, v in per_head[h].items()}
        d["prog_fid_mean"] -= base
        d["prog_fid_removed"] -= base
        res["heads"][str(h)] = d
        print(f"{h:>4} {d['model_self']:>18.4f} {d['model_ctx']:>26.4f} {d['prog_self']:>28.4f} "
              f"{d['prog_fid_mean']:>25.4f} {d['prog_fid_removed']:>24.4f}")
    out = os.path.join(REPO, "analysis", "out", os.path.basename(args.model), f"consistency_{res['program_sha']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
