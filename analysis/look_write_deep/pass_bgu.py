"""The corpus-sample pass: per head and pair slot (d, k, q), accumulate the bigram-stage usefulness of the head's
write (TOTAL and DV readouts), its model-echo part (MGU), a value-shuffled control, the oracle score (OQC) and the
count N, over a sample of aligned 256-token TRAIN windows (the fit and eval windows excluded). Readouts: total
(exact gradient), ltotal (later blocks exact, final LayerNorm linearised at the typical scale; equals dv for a
one-block model), dv (direct path only); echo / lecho: the model-echo versions of total / ltotal.

    .venv/bin/python -m analysis.look_write_deep.pass_bgu attn2L_s1 [--tokens 200000000] [--device mps] [--bs 32]

Per batch: one forward pass with the residuals tracked, two backward passes (true next token; model echo), the DV
readout in closed form. Instance values are gathered on the device and summed into float32 slot accumulators, which
are flushed into float64 every FLUSH batches. Writes analysis/out/look_write_deep/<model>/bgu.npz:
    keys (n_slots,) int64 sorted pair keys (d, k, q); N (n_slots,); and (L, H, n_slots) float64 sums for
    total, dv, echo, shuf, oqc (divide by N + 20 for BGU-type tables, by N for the oracle mean); plus the token-level
    write table obar (L, H, V, D) used by the shuffled control, and the typical final-LN scale.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.look_write_deep import common_deep as C

QUANTITIES = ("total", "ltotal", "dv", "echo", "lecho", "shuf", "lshuf", "oqc")
FLUSH = 50


def sample_window_indices(n_windows_needed, exclude):
    from src.train.data import load_stream

    n_all = (len(load_stream("train")) - 1) // C.T
    order = np.random.default_rng(C.SAMPLE_SEED).permutation(n_all)
    order = order[~np.isin(order, exclude)]
    return order[:n_windows_needed]


def window_indices(seed, n):
    from src.train.data import load_stream

    n_all = (len(load_stream("train")) - 1) // C.T          # analysis.fitlib.train_windows uses the same windows
    return np.sort(np.random.default_rng(seed).choice(n_all, size=n, replace=False))


def token_writes(model, idx, stream, device, n_windows=4000, bs=32):
    """obar[l, h, k]: mean write o_h(j) over positions with t_j = k (token-level write), from n_windows sample
    windows; tokens never seen get the mean write."""
    L, H, DH, D = C.dims(model)
    S = torch.zeros(L, H, C.V, D, dtype=torch.float64)
    cnt = torch.zeros(C.V, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, min(n_windows, len(idx)), bs):
            x = torch.as_tensor(np.stack([np.asarray(stream[i * C.T:i * C.T + C.T], np.int64)
                                          for i in idx[s:s + bs]])).to(device)
            _, info = C.forward(model, x)
            tok = x.reshape(-1).cpu()
            cnt += torch.bincount(tok, minlength=C.V).double()
            for l in range(L):
                for h in range(H):
                    o = C.head_writes(model, info, l, h).reshape(-1, D).double().cpu()
                    S[l, h].index_add_(0, tok, o)
    seen = cnt > 0
    obar = S / cnt.clamp(min=1)[None, None, :, None]
    mean = (S.sum(2) / cnt.sum())[:, :, None, :]
    obar[:, :, ~seen] = mean.expand(-1, -1, int((~seen).sum()), -1)
    return obar.float()


def run(model_id, n_tokens, device, bs):
    from src.train.data import load_stream

    C.check_disk(10)
    t0 = time.time()
    dev = torch.device(device)
    m = C.load(model_id).to(dev)
    L, H, DH, D = C.dims(m)
    stream = load_stream("train")
    fit_idx, eval_idx = window_indices(C.FIT_SEED, C.N_FIT), window_indices(C.EVAL_SEED, C.N_EVAL)
    fit_w, eval_w = C.windows(C.FIT_SEED, C.N_FIT), C.windows(C.EVAL_SEED, C.N_EVAL)
    n_win = n_tokens // C.T
    idx = sample_window_indices(n_win, np.concatenate([fit_idx, eval_idx]))
    keys = C.build_slots(model_id, fit_w, eval_w, n_win * C.T)
    S = len(keys)
    scale = C.final_ln_scale(m.cpu(), fit_w[:, :C.T])
    m = m.to(dev)
    obar = token_writes(m, idx, stream, dev).to(dev)
    perm = torch.as_tensor(np.random.default_rng(0).permutation(C.V), device=dev)
    P1 = torch.as_tensor(C.p1_table(), dtype=torch.float32, device=dev)
    acc = {q: np.zeros((L, H, S)) for q in QUANTITIES}
    N = np.zeros(S)
    part = torch.zeros(len(QUANTITIES), L, H, S, dtype=torch.float32, device=dev)
    part_n = torch.zeros(S, dtype=torch.float32, device=dev)
    b_i, i_i, j_i, d_i = C.pair_instances(np.zeros((bs, C.T), np.int64))
    tb, ti, tj = (torch.as_tensor(a, device=dev) for a in (b_i, i_i, j_i))
    n_batches = (len(idx) + bs - 1) // bs
    for nb, s in enumerate(range(0, len(idx), bs)):
        chunk = idx[s:s + bs]
        w = np.stack([np.asarray(stream[i * C.T:i * C.T + C.T + 1], np.int64) for i in chunk])
        B = len(w)
        if B < bs:                                  # last partial batch: rebuild the instance index
            b_i, i_i, j_i, d_i = C.pair_instances(np.zeros((B, C.T), np.int64))
            tb, ti, tj = (torch.as_tensor(a, device=dev) for a in (b_i, i_i, j_i))
        t = w[:, :C.T]
        slots = C.slot_of(keys, C.pair_key(d_i, t[b_i, j_i], t[b_i, i_i]))
        ok = torch.as_tensor(slots >= 0, device=dev)
        sl = torch.as_tensor(slots, device=dev)[ok]
        x = torch.as_tensor(t, device=dev)
        y = torch.as_tensor(w[:, 1:], device=dev)
        R = C.bigram_error(x, y, P1)
        g_dv = C.direct_readout(m, R, scale)
        with torch.enable_grad():
            logits, info = C.forward(m, x, track=True)
            g_true = torch.autograd.grad((logits * R).sum(), info["resid"], retain_graph=True)
            R_echo = C.bigram_error(x, y, P1, logits=logits)
            g_echo = torch.autograd.grad((logits * R_echo).sum(), info["resid"], retain_graph=True)
            # LTOTAL: later blocks exact, final LayerNorm linearised at the typical scale (= DV for one block)
            g_lt = torch.autograd.grad(info["resid"][-1], info["resid"], grad_outputs=g_dv, retain_graph=True)
            g_le = torch.autograd.grad(info["resid"][-1], info["resid"],
                                       grad_outputs=C.direct_readout(m, R_echo, scale))
        tok_j = x[tb, tj][ok]
        for l in range(L):
            sc = info["scores"][l].detach()
            for h in range(H):
                o = C.head_writes(m, info, l, h)                          # (B, T, D)
                o_shuf = obar[l, h][perm[x]]                             # write of the shuffled token
                vals = {
                    "total": (g_true[l][tb, ti] * o[tb, tj]).sum(-1),
                    "ltotal": (g_lt[l][tb, ti] * o[tb, tj]).sum(-1),
                    "dv": (g_dv[tb, ti] * o[tb, tj]).sum(-1),
                    "echo": (g_echo[l][tb, ti] * o[tb, tj]).sum(-1),
                    "lecho": (g_le[l][tb, ti] * o[tb, tj]).sum(-1),
                    "shuf": (g_true[l][tb, ti] * o_shuf[tb, tj]).sum(-1),
                    "lshuf": (g_lt[l][tb, ti] * o_shuf[tb, tj]).sum(-1),
                    "oqc": sc[tb, h, ti, tj] - sc[tb, h, ti, ti],
                }
                for qi, q in enumerate(QUANTITIES):
                    part[qi, l, h].index_add_(0, sl, vals[q][ok].float())
        part_n.index_add_(0, sl, torch.ones_like(sl, dtype=torch.float32))
        del tok_j
        if (nb + 1) % FLUSH == 0 or nb == n_batches - 1:
            p = part.double().cpu().numpy()
            for qi, q in enumerate(QUANTITIES):
                acc[q] += p[qi]
            N += part_n.double().cpu().numpy()
            part.zero_()
            part_n.zero_()
            if (nb + 1) % (FLUSH * 20) == 0 or nb == n_batches - 1:
                print(f"{model_id}: {(nb + 1) * bs * C.T / 1e6:.0f}M tokens, {time.time() - t0:.0f}s", flush=True)
    out = os.path.join(C.model_dir(model_id), "bgu.npz")
    np.savez(out, keys=keys, N=N, scale=scale, obar=obar.cpu().numpy(), n_tokens=n_win * C.T,
             **{q: acc[q] for q in QUANTITIES})
    meta = {"model": model_id, "n_tokens": int(n_win * C.T), "n_slots": int(S), "scale": scale,
            "seconds": time.time() - t0, "device": device, "batch": bs}
    with open(os.path.join(C.model_dir(model_id), "bgu_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(2)
    for mid in a.models:
        run(mid, a.tokens, a.device, a.bs)
