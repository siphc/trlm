"""Shared pieces for the pre-registered test "where a head looks comes from what it writes"
(logs/prereg_look_follows_write.md). Written independently of analysis/investigations/qk/.

Conventions
- Models: one-block attention-only TinyLM (src/train/model.py), fp32 weights, everything here in float64 numpy
  unless stated otherwise.
- Pair tables are indexed [k, q]: k = key token (earlier), q = query token (later), except TT which the
  pre-registration writes as TT(q, k); tt_tables returns it indexed [q, k] and callers transpose.
- Logit units: OV tables are divided by the model's typical final-LayerNorm scale (final_ln_scale).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "analysis", "out", "look_write")
V, T, H, DH, D = 2000, 256, 4, 32, 128
EPS = 1e-5                        # nn.LayerNorm default eps (checked against the model in selftest.py)
FIT_SEED = 2024                   # train_windows(32, 2024): fitting the attention rules, LOCAL test, LN scale
EVAL_SEED = 2025                  # train_windows(32, 2025): output test (exact-model substitution)
N_WIN = 32
SHRINK = 20.0                     # UOV shrinkage (S - N B) / (N + 20)
D_MAX = 7                         # pair features use distances 1..7
PERM_SEED = 0                     # token-shuffled control: rows of OV_h permuted with default_rng(0)
POS_SET = (16, 48, 96, 160, 224)  # query positions for the pure-pair score at d = 1
MIN_COUNT = 20                    # pure-pair correlation uses pairs with N_1 >= 20
LOCAL_MAX_DIST = 10.0             # LOCAL head: mean attention distance to earlier tokens <= 10
CHUNK = 10_000_000                # corpus positions per chunk (<= 20M by the resource rules)

# distance buckets for j < i (distance 0 is the self term, covered by self x query shape)
BUCKET_EDGES = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 31), (32, 63), (64, 10_000)]
BUCKET_NAMES = ["1", "2", "3", "4", "5-7", "8-15", "16-31", "32-63", "64+"]


def set_threads():
    import torch

    torch.set_num_threads(2)


def load(model_id):
    from harness.common import load_model

    m = load_model(model_id)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def weights(model):
    """Model weights as float64 numpy arrays, per head where relevant."""
    blk = model.blocks[0]
    W = blk.attn.qkv.weight.detach().double().numpy()           # (3D, D): q = x @ W[:D].T, k, v likewise
    Wo = blk.attn.out.weight.detach().double().numpy()          # (D, D): out = z @ Wo.T
    w = {
        "wte": model.wte.weight.detach().double().numpy(),      # (V, D)
        "wpe": model.wpe.weight.detach().double().numpy(),      # (T, D)
        "g1": blk.ln1.weight.detach().double().numpy(),
        "gf": model.ln_f.weight.detach().double().numpy(),
        "Wq": [W[0 * D + h * DH: 0 * D + (h + 1) * DH].T for h in range(H)],   # (D, DH)
        "Wk": [W[1 * D + h * DH: 1 * D + (h + 1) * DH].T for h in range(H)],
        "Wv": [W[2 * D + h * DH: 2 * D + (h + 1) * DH].T for h in range(H)],
        "Wo": [Wo[:, h * DH:(h + 1) * DH].T for h in range(H)],               # (DH, D)
        "mlp": blk.mlp is not None,
    }
    if blk.ln1.eps != EPS or model.ln_f.eps != EPS:
        raise ValueError("LayerNorm eps differs from 1e-5")
    return w


def layernorm(x, g):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * g


def pbar(w):
    """Mean position embedding over positions 1..255."""
    return w["wpe"][1:T].mean(0)


def train_windows(seed):
    from analysis.fitlib import train_windows as tw

    return tw(N_WIN, seed)[:, :T]


# ----------------------------------------------------------------------------- exact forward pieces (torch)

def attention_inputs(model, x):
    """x (B, T) int64 tensor -> x0 (B,T,D), q, k, v (B,H,T,DH), scores (B,H,T,T) causal-masked."""
    import torch

    blk = model.blocks[0]
    B, TT = x.shape
    x0 = model.wte(x) + model.wpe(torch.arange(TT))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
    q = q.view(B, TT, H, DH).transpose(1, 2)
    k = k.view(B, TT, H, DH).transpose(1, 2)
    v = v.view(B, TT, H, DH).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
    sc = sc.masked_fill(~torch.ones(TT, TT, dtype=torch.bool).tril(), float("-inf"))
    return x0, q, k, v, sc


def forward_with_pattern(model, x, head=None, pattern=None):
    """Exact logits (B, T, V) with head `head`'s attention pattern replaced by `pattern` (B, T, T) (rows sum to 1,
    zero above the diagonal). Everything else, including that head's values and output projection, is exact.
    head=None gives the unmodified explicit forward pass."""
    import torch

    blk = model.blocks[0]
    x0, q, k, v, sc = attention_inputs(model, x)
    a = sc.softmax(-1)
    if head is not None:
        a = a.clone()
        a[:, head] = torch.as_tensor(pattern, dtype=a.dtype)
    B, TT = x.shape
    z = (a @ v).transpose(1, 2).reshape(B, TT, D)
    h = x0 + blk.attn.out(z)
    if blk.mlp is not None:
        h = h + blk.mlp(blk.ln2(h))
    return model.ln_f(h) @ model.wte.weight.T


def patterns(model, windows, bs=8):
    """Model attention patterns (n, H, T, T) float32 for windows (n, T)."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[i:i + bs], dtype=np.int64))
            out.append(attention_inputs(model, x)[4].softmax(-1).numpy())
    return np.concatenate(out)


def final_ln_scale(model, windows, bs=8):
    """Typical final-LN scale: mean over all positions of sqrt(var + eps) of the final LN input."""
    import torch

    blk = model.blocks[0]
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[i:i + bs], dtype=np.int64))
            x0, q, k, v, sc = attention_inputs(model, x)
            z = (sc.softmax(-1) @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], D)
            h = x0 + blk.attn.out(z)
            if blk.mlp is not None:
                h = h + blk.mlp(blk.ln2(h))
            hd = h.double()
            sd = torch.sqrt(hd.var(-1, unbiased=False) + EPS)
            tot += float(sd.sum())
            n += sd.numel()
    return tot / n


def logprobs_sub(model, windows, head=None, pats=None, bs=8):
    """log-probs (n, T, V) float32 with head's pattern substituted (pats: (n, T, T))."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[i:i + bs], dtype=np.int64))
            p = None if head is None else pats[i:i + bs]
            out.append(torch.log_softmax(forward_with_pattern(model, x, head, p).float(), -1).numpy())
    return np.concatenate(out)


def kl(lp_ref, lp):
    """Mean per-token KL(ref || other) over all positions, nats."""
    p = np.exp(lp_ref.astype(np.float64))
    return float((p * (lp_ref.astype(np.float64) - lp.astype(np.float64))).sum(-1).mean())


def kl_per_window(lp_ref, lp):
    """Per-window mean KL (n,), for paired standard errors across windows."""
    p = np.exp(lp_ref.astype(np.float64))
    return (p * (lp_ref.astype(np.float64) - lp.astype(np.float64))).sum(-1).mean(-1)


# ----------------------------------------------------------------------------- circuit tables (numpy)

def ov_tables(w, scale):
    """OV_h(k, v) in logit units, (H, V, V) float32: embed k (E_k + pbar), LN1, W_V^h, W_O^h, remove the mean over
    d_model (final LN centring), times final LN gain, unembed (tied), divided by the typical final-LN scale."""
    x = layernorm(w["wte"] + pbar(w)[None], w["g1"])                 # (V, D)
    out = np.empty((H, V, V), dtype=np.float32)
    for h in range(H):
        o = x @ w["Wv"][h] @ w["Wo"][h]                              # (V, D)
        o = o - o.mean(1, keepdims=True)
        out[h] = ((o * w["gf"]) @ w["wte"].T / scale).astype(np.float32)
    return out


def tt_tables(w):
    """TT_h(q, k) = (LN1(E_q + pbar) W_Q^h) . (LN1(E_k + pbar) W_K^h) / sqrt(32), (H, V, V) float32 indexed [q, k]."""
    x = layernorm(w["wte"] + pbar(w)[None], w["g1"])
    return np.stack([(x @ w["Wq"][h]) @ (x @ w["Wk"][h]).T / math.sqrt(DH) for h in range(H)]).astype(np.float32)


def pure_pair_scores(w):
    """Exact score for key token k at distance 1 before query token q, averaged over query positions POS_SET,
    minus the query's self score at the same position. (H, V, V) float64 indexed [k, q]."""
    out = np.zeros((H, V, V))
    for i in POS_SET:
        xq = layernorm(w["wte"] + w["wpe"][i][None], w["g1"])        # q at position i
        xk = layernorm(w["wte"] + w["wpe"][i - 1][None], w["g1"])    # k at position i - 1
        for h in range(H):
            Q = xq @ w["Wq"][h]
            s = (xk @ w["Wk"][h]) @ Q.T / math.sqrt(DH)                # [k, q]
            self_s = (Q * (xq @ w["Wk"][h])).sum(1) / math.sqrt(DH)    # [q]
            out[h] += s - self_s[None, :]
    return out / len(POS_SET)


def p1_table():
    """P1(v | q) = exp(stats().log_follow), row = q."""
    from analysis.mech_eval import stats

    return np.exp(np.asarray(stats().log_follow, dtype=np.float64))


def token_shapes():
    """0 word start (space + letter), 1 continuation (starts with a letter), 2 other, 3 end of story (token 0)."""
    from analysis.mech_eval import stats

    s = np.asarray(stats().shape, dtype=np.int64)
    assert s[0] == 3 and (s == 3).sum() == 1
    return s


def perm():
    return np.random.default_rng(PERM_SEED).permutation(V)


def out_name(model_id):
    """'attn_s1' -> 'attn_s1'; a checkpoint path such as models/law_pressure/both_l10_s1/final.pt ->
    'law_pressure_both_l10_s1' (its folder relative to models/, with '_' for '/')."""
    p = str(model_id)
    if not p.endswith(".pt"):
        return p
    rel = os.path.relpath(os.path.dirname(os.path.abspath(p)), os.path.join(REPO, "models"))
    return rel.replace(os.sep, "_")


def model_dir(model_id):
    d = os.path.join(OUT, out_name(model_id))
    os.makedirs(d, exist_ok=True)
    return d


def check_disk(min_gb=10.0):
    import shutil

    free = shutil.disk_usage(REPO).free / 1e9
    if free < min_gb:
        raise RuntimeError(f"only {free:.1f} GB free; stopping (rule: >= {min_gb} GB)")
    return free
