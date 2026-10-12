"""A torch "student" whose structure is exactly the replacement program's structure.

For a sequence t_0..t_n and position i the program computes

  score_i[v] = FOLLOW(t_i, v)                                  what usually follows token t_i
             + sum_h  sum_{j<=i}  a^h_ij * RECALL^h(t_j, v)      head h recalls tokens seen earlier
  a^h_ij     = softmax_j( SALIENCE^h[t_j] + PREFER^h[class(t_i), class(t_j)] + RECENCY^h[bucket(i-j)] )
  log q_i    = log_softmax(score_i)

with every token-by-token table written as class defaults plus sparse exceptions:

  FOLLOW(t, v)    = WORD[v] + CLASS_FOLLOW[class(t), class(v)] + EXCEPTION[t, v]
  RECALL^h(s, v)  = CLASS_RECALL^h[class(s), class(v)] + COPY^h[class(s)] * [s == v] + EXCEPTION^h[s, v]

All tables are fitted by minimising KL(model || student) on TRAIN windows (analysis/fitlib.py),
with an L1 penalty on the exceptions so that most of them are exactly zero after pruning.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from analysis.fitlib import Target, train_windows

V, T = 2000, 256
# Recency buckets: distances 0..15 one by one, then wider bands. Bucket b covers [START[b], START[b+1]).
BUCKET_START = list(range(16)) + [16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224]


def bucket_of_distance() -> np.ndarray:
    return np.searchsorted(np.asarray(BUCKET_START), np.arange(T), side="right") - 1


class Student(nn.Module):
    def __init__(self, n_heads: int, class_of: np.ndarray, n_classes: int):
        super().__init__()
        self.H, self.K = n_heads, n_classes
        self.register_buffer("cls", torch.as_tensor(class_of, dtype=torch.long))
        self.word = nn.Parameter(torch.zeros(V))
        self.class_follow = nn.Parameter(torch.zeros(n_classes, n_classes))
        self.follow_exc = nn.Parameter(torch.zeros(V, V))
        self.salience = nn.Parameter(torch.zeros(n_heads, V))
        self.prefer = nn.Parameter(torch.zeros(n_heads, n_classes, n_classes))
        self.recency = nn.Parameter(torch.zeros(n_heads, len(BUCKET_START)))
        self.register_buffer("bucket", torch.as_tensor(bucket_of_distance(), dtype=torch.long))
        self.class_recall = nn.Parameter(torch.zeros(n_heads, n_classes, n_classes))
        self.copy = nn.Parameter(torch.zeros(n_heads, n_classes))
        self.recall_exc = nn.Parameter(torch.zeros(n_heads, V, V))
        # Masks fix pruned exceptions at zero during refits (1 = free, 0 = pruned).
        self.register_buffer("follow_mask", torch.ones(V, V))
        self.register_buffer("recall_mask", torch.ones(n_heads, V, V))

    def follow(self) -> torch.Tensor:
        c = self.cls
        return self.word[None, :] + self.class_follow[c][:, c] + self.follow_exc * self.follow_mask

    def recall(self, h: int) -> torch.Tensor:
        c = self.cls
        return (self.class_recall[h][c][:, c] + torch.diag(self.copy[h][c])
                + self.recall_exc[h] * self.recall_mask[h])

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, T) attention weights."""
        n = x.shape[1]
        c = self.cls[x]
        sal = self.salience[:, x].permute(1, 0, 2)                         # (B, H, T)
        pref = self.prefer[:, c[:, :, None], c[:, None, :]].permute(1, 0, 2, 3)
        ar = torch.arange(n, device=x.device)
        dist = ar[:, None] - ar[None, :]
        rec = self.recency[:, self.bucket[dist.clamp(min=0)]]
        s = sal[:, :, None, :] + pref + rec[None]
        s = s.masked_fill(dist < 0, float("-inf"))
        return torch.softmax(s, -1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        a = self.attention(x)
        score = self.follow()[x]
        for h in range(self.H):
            R = self.recall(h)
            score = score + a[:, h] @ R[x]
        lp = torch.log_softmax(score, -1)
        return (lp, a) if return_attn else lp

    def l1(self) -> torch.Tensor:
        return (self.follow_exc * self.follow_mask).abs().sum() + (self.recall_exc * self.recall_mask).abs().sum()

    def prune(self, thresh: float) -> dict:
        with torch.no_grad():
            self.follow_mask *= (self.follow_exc.abs() >= thresh).float()
            self.recall_mask *= (self.recall_exc.abs() >= thresh).float()
            self.follow_exc *= self.follow_mask
            self.recall_exc *= self.recall_mask
        return {"follow_exceptions": int(self.follow_mask.sum()),
                "recall_exceptions": [int(self.recall_mask[h].sum()) for h in range(self.H)]}


def prune_to_budget(st: Student, n_follow: int, n_recall: int, token_freq: np.ndarray) -> dict:
    """Keep the n_follow AFTER exceptions and n_recall RECALL exceptions per head with the largest
    importance |value| * sqrt(frequency of the first token); zero and freeze the rest."""
    w = torch.as_tensor(np.sqrt(token_freq / token_freq.sum()), dtype=torch.float32, device=st.cls.device)
    with torch.no_grad():
        def keep(p, m, n):
            imp = (p * m).abs() * w[:, None]
            if n >= imp.numel():
                return m
            thr = torch.topk(imp.flatten(), n).values[-1] if n > 0 else torch.tensor(float("inf"))
            return m * (imp >= thr).float() * (imp > 0).float()
        st.follow_mask.copy_(keep(st.follow_exc, st.follow_mask, n_follow))
        for h in range(st.H):
            st.recall_mask[h].copy_(keep(st.recall_exc[h], st.recall_mask[h], n_recall))
        st.follow_exc *= st.follow_mask
        st.recall_exc *= st.recall_mask
    return {"follow_exceptions": int(st.follow_mask.sum()),
            "recall_exceptions": [int(st.recall_mask[h].sum()) for h in range(st.H)]}


def model_patterns(tgt: Target, x: torch.Tensor) -> torch.Tensor:
    """The model's own attention weights, (B, H, T, T)."""
    with torch.no_grad():
        _, pats = tgt.model(x, return_patterns=True)
    return pats[0]


def fit_attention(st: Student, tgt: Target, W: torch.Tensor, steps=1500, bs=16, lr=0.05, seed=0,
                  log_every=100):
    """Stage 1: fit each student head's attention (SALIENCE, PREFER, RECENCY) to the matching model
    head's attention pattern, by KL(model pattern || student pattern) averaged over query positions."""
    params = [st.salience, st.prefer, st.recency]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed + 2)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(W), (bs,), generator=g)
        x = W[idx, :T].to(tgt.device)
        P = model_patterns(tgt, x)
        A = st.attention(x)
        kl = (P * (torch.log(P + 1e-12) - torch.log(A + 1e-12))).sum(-1).mean((0, 2))   # per head
        loss = kl.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"attn step {step} pattern KL per head {[round(v, 3) for v in kl.tolist()]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return kl.tolist()


def shrink(st: Student, tau: float) -> None:
    """Proximal L1 step: soft-threshold the exception tables so small entries become exactly 0."""
    with torch.no_grad():
        for p, m in ((st.follow_exc, st.follow_mask), (st.recall_exc, st.recall_mask)):
            p.copy_(torch.sign(p) * torch.clamp(p.abs() - tau, min=0.0) * m)


def fit(st: Student, model_id: str, n_windows=6000, steps=3000, bs=16, lr=0.01, l1=0.0, device="mps",
        log_every=100, seed=0, tgt: Target | None = None, W=None, tau=0.0, freeze_attention=False):
    torch.manual_seed(seed)
    tgt = tgt or Target(model_id, device)
    W = W if W is not None else torch.as_tensor(train_windows(n_windows))
    st.to(device)
    params = [p for n, p in st.named_parameters()
              if not (freeze_attention and n in ("salience", "prefer", "recency"))]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    run = 0.0
    for step in range(steps):
        idx = torch.randint(0, len(W), (bs,), generator=g)
        x = W[idx, :T].to(device)
        p_log = tgt.logprobs(x)
        q_log = st(x)
        kl = (p_log.exp() * (p_log - q_log)).sum(-1).mean()
        loss = kl + (l1 * st.l1() / (V * V) if l1 else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if tau:
            shrink(st, tau)
        run = 0.98 * run + 0.02 * kl.item() if step else kl.item()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "kl": kl.item(), "kl_avg": run, "t": time.time() - t0})
            print(f"step {step} kl {kl.item():.4f} avg {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return hist


@torch.no_grad()
def evaluate(st: Student, tgt: Target, W: torch.Tensor, bs: int = 16) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(W), bs):
        x = W[i : i + bs, :T].to(tgt.device)
        p_log = tgt.logprobs(x)
        tot += float((p_log.exp() * (p_log - st(x))).sum(-1).sum())
        n += x.numel()
    return tot / n
