"""Study B, "law_pressure" (logs/plan_attention_law_and_readability.md section 5; pre-registration
logs/prereg_law_pressure.md): a training penalty against the look-follows-write law.

For every head h and every pair (query i, key j = i - d, d = 1..7) of the training batch, queries i = 7..T-1:
    s_ij  the raw attention score q_i . k_j / sqrt(d_head)
    u_ij  the instance usefulness of the head's own vote from t_j for the token y_i that actually comes next, beyond
          the bigram: OV_h(t_j, y_i) - sum_v P1(v | t_i) OV_h(t_j, v).
          OV_h(s, v) = A_h(s) . E(v), with A_h(s) = s embedded at the mean position, LN1, W_V^h W_O^h, centred over
          d_model, times the final LayerNorm gain, and E the tied embedding (as src/train/readable.ov_tables_torch), so
          u_ij = A_h(t_j) . (E(y_i) - (P1 E)(t_i)) and nothing V x V is formed per pair.
Both are double-centred (over each query's 7 keys, then over each distance), and the penalty is lambda times the mean
over heads of the squared correlation of s and u, pooled over the batch. Arms: "both" (the penalty's gradient reaches
where heads look and what they write), "attn" (u detached: only attention may move away from the law), "vote"
(s detached: only the votes may change). P1 is FOLLOW's bigram rule ('follow' in data/fixed_votes/tables.npz).

Nothing here consumes random numbers, so a lambda = 0 run is bitwise the plain run (src/train/test_sanity_law_pressure.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

D_MAX = 7
ARMS = ("both", "attn", "vote")


def vote_vectors(model) -> torch.Tensor:
    """A_h(s) for every head and token, (H, V, D); differentiable. OV_h = A_h @ E.T."""
    blk = model.blocks[0]
    D, Dh, H = model.cfg.d_model, model.cfg.d_head, model.cfg.n_heads
    T = model.cfg.context_length
    X = blk.ln1(model.wte.weight + model.wpe.weight[1:T].mean(0))
    Wv, Wo = blk.attn.qkv.weight[2 * D: 3 * D], blk.attn.out.weight
    out = []
    for h in range(H):
        o = (X @ Wv[h * Dh:(h + 1) * Dh].T) @ Wo[:, h * Dh:(h + 1) * Dh].T
        out.append((o - o.mean(1, keepdim=True)) * model.ln_f.weight)
    return torch.stack(out)


def law_terms(model, x: torch.Tensor, p1: torch.Tensor):
    """s, u: (H, B, Q, D_MAX) for queries i = D_MAX..T-1 of the inputs x[:, :T], next tokens x[:, 1:]; column d-1 holds
    the key d tokens back. p1: (V, V) bigram probabilities, row = current token."""
    blk = model.blocks[0]
    D, Dh, H = model.cfg.d_model, model.cfg.d_head, model.cfg.n_heads
    B, T = x.shape[0], x.shape[1] - 1
    inp = x[:, :T]
    x0 = model.wte(inp) + model.wpe(torch.arange(T, device=x.device))[None]
    q, k, _ = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
    q = q.view(B, T, H, Dh).permute(2, 0, 1, 3)                       # (H, B, T, Dh)
    k = k.view(B, T, H, Dh).permute(2, 0, 1, 3)
    qq = q[:, :, D_MAX:]
    s = torch.stack([(qq * k[:, :, D_MAX - d:T - d]).sum(-1) for d in range(1, D_MAX + 1)], -1) / math.sqrt(Dh)
    # gathers use F.embedding: its backward is deterministic on CPU (advanced indexing's is not; resume test)
    A = vote_vectors(model)                                           # (H, V, D)
    E = model.wte.weight
    r = F.embedding(x[:, D_MAX + 1:T + 1], E) - F.embedding(inp[:, D_MAX:], p1 @ E)      # (B, Q, D)
    u = torch.stack([torch.stack([(F.embedding(inp[:, D_MAX - d:T - d], A[h]) * r).sum(-1) for h in range(H)])
                     for d in range(1, D_MAX + 1)], -1)
    return s, u


def double_centre(z: torch.Tensor) -> torch.Tensor:
    """Remove each query's mean over its keys, then each distance's mean (per head); exact for this balanced layout."""
    z = z - z.mean(-1, keepdim=True)
    return z - z.mean((1, 2), keepdim=True)


def head_correlations(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Pooled correlation per head of the double-centred s and u, (H,)."""
    s, u = double_centre(s), double_centre(u)
    return (s * u).sum((1, 2, 3)) / ((s * s).sum((1, 2, 3)) * (u * u).sum((1, 2, 3))).clamp_min(1e-30).sqrt()


class LawPenalty:
    def __init__(self, cfg, device):
        assert cfg.law_arm in ARMS, cfg.law_arm
        z = np.load(cfg.law_p1_path)
        self.p1 = torch.as_tensor(np.exp(z["follow"].astype(np.float64)), dtype=torch.float32, device=device)
        self.p1_sha = str(z["sha"])
        self.arm, self.lam = cfg.law_arm, cfg.lambda_law
        self.ramp = int(round(cfg.law_ramp * cfg.total_steps))

    def summary(self) -> dict:
        return {"arm": self.arm, "lambda": self.lam, "ramp_steps": self.ramp, "p1_sha": self.p1_sha}

    def scale(self, step: int) -> float:
        return 1.0 if self.ramp <= 0 else min(1.0, (step + 1) / self.ramp)

    def loss(self, model, x: torch.Tensor, step: int):
        """The weighted penalty on the training batch x (B, T + 1) and the per-head correlations for logging."""
        s, u = law_terms(model, x, self.p1)
        if self.arm == "attn":
            u = u.detach()
        elif self.arm == "vote":
            s = s.detach()
        rho = head_correlations(s.float(), u.float())
        info = {f"law_rho_h{h}": rho[h].detach() for h in range(len(rho))}
        return self.scale(step) * self.lam * (rho ** 2).mean(), info


@torch.no_grad()
def corr_on(model, windows: np.ndarray, p1: torch.Tensor, bs: int = 16) -> list[float]:
    """The in-batch correlation per head pooled over fixed windows (n, T + 1), centred over all of them (offline
    measurement of the trained quantity; windows from analysis.fitlib.train_windows)."""
    parts = [law_terms(model, torch.as_tensor(np.asarray(windows[i:i + bs], np.int64), device=p1.device), p1)
             for i in range(0, len(windows), bs)]
    s = torch.cat([p[0] for p in parts], 1).double()
    u = torch.cat([p[1] for p in parts], 1).double()
    return head_correlations(s, u).tolist()
