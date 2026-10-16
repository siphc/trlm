"""Shared pieces for the deep study of "where a head looks comes from what it writes" (logs/decisions.md item 8;
plan: logs/deep_study.md). Works for attention-only TinyLM models with any number of blocks, including the one-block
models of the main track (for calibration against analysis/look_write/).

Definitions (see logs/prereg_deep_look_write.md):
- Bigram-stage error at position i: R_i = e_{y_i} - P1(. | t_i)  (P1: the program's corpus bigram rule).
  F = sum_i <logits_i, R_i>.  g_i^(l) = dF / d(residual after block l) at position i (all paths; "TOTAL").
- A head's write from key j: o_h(j) = v_h(j) W_O^h (the vector it adds to the residual stream when it attends fully
  to j).  u(j -> i) = o_h(j) . g_i^(l).  BGU_h,d(k, q) = sum of u over sample instances (t_j = k, t_i = q,
  i - j = d) / (N + 20).
- Readouts: TOTAL (above); DV (direct path only: g = centre(gf * (R @ W_E)) / typical final-LN scale, the one-layer
  definition); SAME (keys and values of later blocks computed from a detached residual, so only the same position's
  own path counts; approximation, also removes the self value path of later blocks).
- MGU (model echo): R replaced by softmax(logits) - P1; CGU = BGU - MGU (the model's own-loss gradient part).
- OQC (oracle ceiling): mean over instances of the head's own score s_ij - s_ii, per (k, q, d).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from analysis.look_write import common as C1  # noqa: E402  (shared constants and helpers of the one-layer study)

OUT = os.path.join(REPO, "analysis", "out", "look_write_deep")
V, T, EPS, SHRINK, D_MAX = C1.V, C1.T, C1.EPS, C1.SHRINK, C1.D_MAX
FIT_SEED, EVAL_SEED = 2024, 2025
N_FIT, N_EVAL = 32, 64
SAMPLE_SEED = 7                    # order of the TRAIN windows used for the corpus-sample pass
layernorm, kl, kl_per_window, p1_table, token_shapes, check_disk = (C1.layernorm, C1.kl, C1.kl_per_window,
                                                                    C1.p1_table, C1.token_shapes, C1.check_disk)


# ----------------------------------------------------------------------------- models

def model_path(model_id):
    """'attn2L_s0' -> models/deep/attn2L_s0/final.pt; 'attn2L_s0@ckpt_step229' -> that checkpoint;
    one-block main-track ids ('attn_s1', 'attn_s1@ckpt_step718') -> models/final/...; existing paths unchanged."""
    if os.path.exists(model_id):
        return model_id
    name, _, ckpt = model_id.partition("@")
    root = "deep" if "L_s" in name else "final"
    return os.path.join(REPO, "models", root, name, (ckpt or "final") + ".pt")


def load(model_id, device="cpu"):
    from src.train.train import load_model

    m = load_model(model_path(model_id), device)
    for p in m.parameters():
        p.requires_grad_(False)
    if any(b.mlp is not None for b in m.blocks):
        raise ValueError("deep study is for attention-only models")
    return m


def dims(model):
    return len(model.blocks), model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model


def head_out_weight(model, layer, head):
    """W_O^h as (DH, D): o = v_h @ W."""
    _, H, DH, D = dims(model)
    return model.blocks[layer].attn.out.weight[:, head * DH:(head + 1) * DH].T


def windows(seed, n):
    from analysis.fitlib import train_windows

    return train_windows(n, seed)[:, :T + 1]


# ----------------------------------------------------------------------------- explicit forward

def forward(model, x, sub=None, frozen=None, detach_kv_after=None, track=False, ablate_ctx=None):
    """Explicit forward pass of a (multi-block, attention-only) TinyLM.

    x: (B, T') int64.  sub: (layer, head, pattern) replaces that head's attention pattern (B, T', T') in that block;
    later blocks recompute their patterns from the modified residual unless `frozen` gives them ({layer: (B,H,T,T)}).
    detach_kv_after: in blocks after this layer, keys and values are computed from a detached residual (SAME readout).
    track: the embedding output requires grad, so gradients w.r.t. the residuals can be taken.
    ablate_ctx: {(layer, head): mean_ctx (DH,)}: that head's output from earlier tokens is replaced by mean_ctx, keeping
    its current-token part: z_h(i) = a_ii v_i + mean_ctx (the harness "earlier" ablation, at any layer).
    Returns logits (B, T', V) and a dict with per-block residuals (resid[l] = after block l; resid[-1]... see code),
    scores, patterns and values."""
    L, H, DH, D = dims(model)
    B, TT = x.shape
    h = model.wte(x) + model.wpe(torch.arange(TT, device=x.device))[None]
    if track:
        h = h.detach().requires_grad_(True)
    info = {"embed": h, "resid": [], "scores": [], "pattern": [], "v": []}
    mask = torch.ones(TT, TT, dtype=torch.bool, device=x.device).tril()
    for l, blk in enumerate(model.blocks):
        xin = blk.ln1(h)
        W = blk.attn.qkv.weight
        xkv = xin.detach() if (detach_kv_after is not None and l > detach_kv_after) else xin
        q = (xin @ W[:D].T).view(B, TT, H, DH).transpose(1, 2)
        k = (xkv @ W[D:2 * D].T).view(B, TT, H, DH).transpose(1, 2)
        v = (xkv @ W[2 * D:].T).view(B, TT, H, DH).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
        sc = sc.masked_fill(~mask, float("-inf"))
        a = frozen[l] if (frozen is not None and l in frozen) else sc.softmax(-1)
        if sub is not None and sub[0] == l:
            a = a.clone()
            a[:, sub[1]] = torch.as_tensor(sub[2], dtype=a.dtype, device=a.device)
        zh = a @ v                                                   # (B, H, T, DH)
        if ablate_ctx:
            for (la, ha), mean_ctx in ablate_ctx.items():
                if la == l:
                    zh = zh.clone()
                    zh[:, ha] = torch.diagonal(a[:, ha], dim1=-2, dim2=-1)[..., None] * v[:, ha] + mean_ctx
        z = zh.transpose(1, 2).reshape(B, TT, D)
        h = h + blk.attn.out(z)
        info["resid"].append(h)
        info["scores"].append(sc)
        info["pattern"].append(a)
        info["v"].append(v)
    logits = model.ln_f(h) @ model.wte.weight.T
    return logits, info


def patterns(model, wins, bs=8):
    """Attention patterns, list over layers of (n, H, T, T) float32 numpy, for windows (n, >= T)."""
    out = None
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs, :T], dtype=np.int64))
            _, info = forward(model, x)
            ps = [p.numpy().astype(np.float32) for p in info["pattern"]]
            out = [[p] for p in ps] if out is None else [o + [p] for o, p in zip(out, ps)]
    return [np.concatenate(o) for o in out]


def logprobs(model, wins, sub=None, frozen_from_clean=False, bs=8):
    """log-probs (n, T, V) float32; sub = (layer, head, patterns (n, T, T)) substituted in that block.
    frozen_from_clean: later blocks keep the clean model's patterns (diagnostic `freeze_later`)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs, :T], dtype=np.int64))
            s = None if sub is None else (sub[0], sub[1], sub[2][i:i + bs])
            frozen = None
            if s is not None and frozen_from_clean:
                _, clean = forward(model, x)
                frozen = {l: clean["pattern"][l] for l in range(s[0] + 1, len(model.blocks))}
            logits, _ = forward(model, x, sub=s, frozen=frozen)
            out.append(torch.log_softmax(logits.float(), -1).numpy())
    return np.concatenate(out)


def ctx_means(model, wins, bs=8):
    """Mean over windows and positions of each head's output from earlier tokens, z_h(i) - a_ii v_i:
    {(layer, head): (DH,) tensor}."""
    L, H, DH, D = dims(model)
    acc = torch.zeros(L, H, DH, dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs, :T], dtype=np.int64))
            _, info = forward(model, x)
            for l in range(L):
                a, v = info["pattern"][l], info["v"][l]
                ctx = a @ v - torch.diagonal(a, dim1=-2, dim2=-1)[..., None] * v      # (B, H, T, DH)
                acc[l] += ctx.double().sum((0, 2))
            n += x.numel()
    return {(l, h): (acc[l, h] / n).float() for l in range(L) for h in range(H)}


def final_ln_scale(model, wins, bs=8):
    """Typical final-LN scale: mean over positions of sqrt(var + eps) of the final LN input."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            x = torch.as_tensor(np.asarray(wins[i:i + bs, :T], dtype=np.int64))
            _, info = forward(model, x)
            h = info["resid"][-1].double()
            sd = torch.sqrt(h.var(-1, unbiased=False) + EPS)
            tot += float(sd.sum())
            n += sd.numel()
    return tot / n


# ----------------------------------------------------------------------------- readouts

def bigram_error(x_in, y, P1, logits=None):
    """R (B, T, V): e_y - P1(. | t_i); with logits given, the model-echo version softmax(logits) - P1."""
    base = P1[x_in]
    if logits is None:
        return torch.nn.functional.one_hot(y, V).to(base.dtype) - base
    return torch.softmax(logits.detach(), -1).to(base.dtype) - base


def direct_readout(model, R, scale):
    """DV gradient (B, T, D) for every layer: centre(gf * (R @ W_E)) / scale (final LN linearised at the typical
    scale, the one-layer definition)."""
    c = (R @ model.wte.weight) * model.ln_f.weight / scale
    return c - c.mean(-1, keepdim=True)


def residual_grads(model, x_in, R, detach_kv_after=None):
    """g^(l) for every block l: d<logits, R>/d(residual after block l), (B, T, D) each; plus the forward info."""
    with torch.enable_grad():
        logits, info = forward(model, x_in, track=True, detach_kv_after=detach_kv_after)
        F = (logits * R).sum()
        grads = torch.autograd.grad(F, info["resid"])
    return [g.detach() for g in grads], logits.detach(), info


def head_writes(model, info, layer, head):
    """o_h(j) for every position j: (B, T, D)."""
    return info["v"][layer][:, head].detach() @ head_out_weight(model, layer, head)


# ----------------------------------------------------------------------------- pair slots

def pair_key(d, k, q):
    return (np.asarray(d, np.int64) * V + np.asarray(k, np.int64)) * V + np.asarray(q, np.int64)


def build_slots(model_tag, fit_w, eval_w, n_sample_tokens, min_expected=20.0):
    """The (d, k, q) pairs whose tables are needed: every pair at d = 1 with expected sample count >= min_expected
    (from the whole-stream counts analysis/out/candidates/qk/N_1.npy scaled to the sample), plus every pair that
    occurs at d = 1..7 in the fit or eval windows. Returns the sorted key array."""
    N1 = np.load(os.path.join(REPO, "analysis", "out", "candidates", "qk", "N_1.npy")).astype(np.float64)
    frac = n_sample_tokens / 527_489_811
    k1, q1 = np.nonzero(N1 * frac >= min_expected)
    keys = [pair_key(1, k1, q1)]
    for w in (fit_w, eval_w):
        t = np.asarray(w[:, :T], dtype=np.int64)
        for d in range(1, D_MAX + 1):
            keys.append(pair_key(d, t[:, :-d].ravel(), t[:, d:].ravel()))
    return np.unique(np.concatenate(keys))


def slot_of(sorted_keys, keys):
    """Slot index of each key (-1 if the key is not stored)."""
    idx = np.searchsorted(sorted_keys, keys)
    idx = np.minimum(idx, len(sorted_keys) - 1)
    return np.where(sorted_keys[idx] == keys, idx, -1)


def pair_instances(t):
    """For token windows t (B, T): flat arrays (b, i, j, d) of all instances with d = i - j in 1..D_MAX."""
    B, TT = t.shape
    bs, iis, jjs, ds = [], [], [], []
    for d in range(1, D_MAX + 1):
        i = np.arange(d, TT)
        bb, ii = np.meshgrid(np.arange(B), i, indexing="ij")
        bs.append(bb.ravel()); iis.append(ii.ravel()); jjs.append((ii - d).ravel()); ds.append(np.full(bb.size, d))
    return [np.concatenate(a) for a in (bs, iis, jjs, ds)]


def model_dir(tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    return d
