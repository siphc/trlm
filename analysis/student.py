"""A torch "student" whose structure is exactly the replacement program's structure.

For a sequence t_0..t_n and position i the program computes

  score_i[v] = FOLLOW(t_i, v)                                  what usually follows token t_i
             + sum_h  sum_{j<=i}  a^h_ij * RECALL^h(t_j, v)      head h recalls tokens seen earlier
  a^h_ij     = softmax_j( SALIENCE^h[t_j] + PREFER^h[att(t_i), att(t_j)] + RECENCY^h[bucket(i-j)] )
  log q_i    = log_softmax(score_i)

with every token-by-token table written as class defaults plus sparse exceptions:

  FOLLOW(t, v)    = WORD[v] + CLASS_FOLLOW[ctx(t), tgt(v)] + EXCEPTION[t, v]
  RECALL^h(s, v)  = CLASS_RECALL^h[ctx(s), tgt(v)] + COPY^h[ctx(s)] * [s == v] + EXCEPTION^h[s, v]

att, ctx and tgt are token -> class maps (buffers cls, ctx, tgt): ATTENTION, CONTEXT and TARGET classes of
program/categories.py. Students of v1-v10 have one taxonomy (ctx = tgt = att); loading their checkpoints sets
ctx and tgt to their cls.

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
    def __init__(self, n_heads: int, class_of: np.ndarray, n_classes: int, self_votes: bool = True,
                 context_of: np.ndarray | None = None, target_of: np.ndarray | None = None):
        """class_of / n_classes: ATTENTION classes (PREFER). context_of / target_of: row / column classes of the
        FOLLOW and RECALL class tables (default class_of: one taxonomy for every role, as in v1-v10)."""
        super().__init__()
        ctx = np.asarray(class_of if context_of is None else context_of)
        tgt = np.asarray(class_of if target_of is None else target_of)
        self.H, self.K = n_heads, n_classes
        self.Kr, self.Kc = int(ctx.max()) + 1, int(tgt.max()) + 1
        # 1.0: heads vote through the current token too; 0.0: a head that looks at the current token
        # casts no vote there (everything the current token says then lives in FOLLOW).
        self.register_buffer("self_votes", torch.tensor(1.0 if self_votes else 0.0))
        self.register_buffer("cls", torch.as_tensor(class_of, dtype=torch.long))
        self.register_buffer("ctx", torch.as_tensor(ctx, dtype=torch.long))
        self.register_buffer("tgt", torch.as_tensor(tgt, dtype=torch.long))
        self.word = nn.Parameter(torch.zeros(V))
        self.class_follow = nn.Parameter(torch.zeros(self.Kr, self.Kc))
        self.follow_exc = nn.Parameter(torch.zeros(V, V))
        self.salience = nn.Parameter(torch.zeros(n_heads, V))
        self.prefer = nn.Parameter(torch.zeros(n_heads, n_classes, n_classes))
        self.recency = nn.Parameter(torch.zeros(n_heads, len(BUCKET_START)))
        self.register_buffer("bucket", torch.as_tensor(bucket_of_distance(), dtype=torch.long))
        self.class_recall = nn.Parameter(torch.zeros(n_heads, self.Kr, self.Kc))
        self.copy = nn.Parameter(torch.zeros(n_heads, self.Kr))
        self.recall_exc = nn.Parameter(torch.zeros(n_heads, V, V))
        # CONFIDENCE: per current token log inverse temperature; all votes at position i are multiplied by
        # exp(log_conf[t_i]) (stands in for the model's final-LayerNorm scale). Zero = neutral.
        self.log_conf = nn.Parameter(torch.zeros(V))
        # Masks fix pruned exceptions at zero during refits (1 = free, 0 = pruned).
        self.register_buffer("follow_mask", torch.ones(V, V))
        self.register_buffer("recall_mask", torch.ones(n_heads, V, V))

    def follow(self) -> torch.Tensor:
        return self.word[None, :] + self.class_follow[self.ctx][:, self.tgt] + self.follow_exc * self.follow_mask

    def recall(self, h: int) -> torch.Tensor:
        return (self.class_recall[h][self.ctx][:, self.tgt] + torch.diag(self.copy[h][self.ctx])
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        sd = dict(state_dict)
        sd.setdefault("self_votes", torch.tensor(1.0))  # checkpoints from before the flag existed
        sd.setdefault("log_conf", torch.zeros(V))       # checkpoints from before CONFIDENCE existed
        sd.setdefault("ctx", sd["cls"])                 # checkpoints from before CONTEXT / TARGET classes:
        sd.setdefault("tgt", sd["cls"])                 # one taxonomy for every role
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def vote_weights(self, a: torch.Tensor) -> torch.Tensor:
        """Attention weights used for voting: the diagonal is removed when heads are silent on self."""
        if float(self.self_votes) == 0.0:
            n = a.shape[-1]
            return a * (1.0 - torch.eye(n, device=a.device, dtype=a.dtype))
        return a

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        a = self.vote_weights(self.attention(x))
        score = self.follow()[x]
        for h in range(self.H):
            R = self.recall(h)
            score = score + a[:, h] @ R[x]
        score = score * torch.exp(self.log_conf[x])[..., None]
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


def follow_importance(st: Student, token_freq: np.ndarray) -> torch.Tensor:
    """Approximate KL cost of dropping each AFTER exception (t, v):
    freq(t) * |w| * max(q_with(v|t), q_without(v|t)), q from the FOLLOW scores with and without w."""
    with torch.no_grad():
        f = st.follow()
        w = st.follow_exc * st.follow_mask
        q_with = torch.softmax(f, -1)
        q_without = torch.softmax(f - w, -1)
        fr = torch.as_tensor(token_freq / token_freq.sum(), dtype=f.dtype, device=f.device)
        return fr[:, None] * w.abs() * torch.maximum(q_with, q_without)


def prune_follow_by_kl(st: Student, n_follow: int, token_freq: np.ndarray) -> int:
    imp = follow_importance(st, token_freq)
    with torch.no_grad():
        if n_follow < int((imp > 0).sum()):
            thr = torch.topk(imp.flatten(), n_follow).values[-1]
            st.follow_mask *= (imp >= thr).float()
        st.follow_mask *= (imp > 0).float()
        st.follow_exc *= st.follow_mask
    return int(st.follow_mask.sum())


@torch.no_grad()
def data_importance(st: Student, W: torch.Tensor, n_batches: int = 40, bs: int = 16, seed: int = 11):
    """Expected first-order cost of dropping each exception, measured with the full program on data.

    FOLLOW (t, v): sum over positions i with t_i = t of q_i(v).
    RECALL^h (s, v): sum over positions i and attended positions j with t_j = s of a^h_ij * q_i(v).
    Importance = |w| * max(S, S * exp(-w)) (a suppressed token would regain mass exp(-w)), clamped.
    """
    dev = st.cls.device
    S_f = torch.zeros(V, V, device=dev)
    S_r = torch.zeros(st.H, V, V, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = W[torch.randint(0, len(W), (bs,), generator=g), :T].to(dev)
        lp, a = st(x, return_attn=True)                                  # a = vote weights
        q = lp.exp()                                                     # (B, T, V)
        S_f.index_add_(0, x.reshape(-1), q.reshape(-1, V))
        for h in range(st.H):
            m = a[:, h].transpose(1, 2) @ q                               # (B, T_j, V)
            S_r[h].index_add_(0, x.reshape(-1), m.reshape(-1, V))

    def imp(w, S):
        return w.abs() * torch.maximum(S, torch.clamp(S * torch.exp(-w), max=S.sum(-1, keepdim=True)))

    return imp(st.follow_exc * st.follow_mask, S_f), torch.stack(
        [imp(st.recall_exc[h] * st.recall_mask[h], S_r[h]) for h in range(st.H)])


@torch.no_grad()
def fisher_importance(st: Student, W: torch.Tensor, n_batches: int = 40, bs: int = 16, seed: int = 11):
    """Second-order cost of dropping each exception at a fitted optimum: 0.5 * w^2 * F, with
    F_FOLLOW(t, v) = sum over positions i with t_i = t of q_i(v) (1 - q_i(v)) and
    F_RECALL^h(s, v) = sum_i sum_{j: t_j = s} (a^h_ij)^2 q_i(v) (1 - q_i(v))   (vote weights a).
    Linear-in-attention rankings overvalue diffuse heads by about 1/a (logs/dead_ends.md, v5)."""
    dev = st.cls.device
    F_f = torch.zeros(V, V, device=dev)
    F_r = torch.zeros(st.H, V, V, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = W[torch.randint(0, len(W), (bs,), generator=g), :T].to(dev)
        lp, a = st(x, return_attn=True)
        q = lp.exp()
        qq = q * (1 - q)
        F_f.index_add_(0, x.reshape(-1), qq.reshape(-1, V))
        for h in range(st.H):
            m = (a[:, h] ** 2).transpose(1, 2) @ qq
            F_r[h].index_add_(0, x.reshape(-1), m.reshape(-1, V))
    wf = st.follow_exc * st.follow_mask
    wr = st.recall_exc * st.recall_mask
    return 0.5 * wf ** 2 * F_f, 0.5 * wr ** 2 * F_r


@torch.no_grad()
def make_nonredundant(st: Student, W: torch.Tensor, n_batches: int = 10, bs: int = 16, seed: int = 13) -> torch.Tensor:
    """Move each head's average vote through the current token into FOLLOW and silence heads on self.
    abar^h(t) = mean self-attention weight of head h when the current token is t (on W); then
    AFTER(t, v) += sum_h abar^h(t) * RECALL^h(t, v). Returns abar (H, V)."""
    dev = st.cls.device
    S = torch.zeros(st.H, V, device=dev)
    C = torch.zeros(V, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = W[torch.randint(0, len(W), (bs,), generator=g), :T].to(dev)
        a = st.attention(x)                                                # (B, H, T, T)
        diag = torch.diagonal(a, dim1=-2, dim2=-1)                         # (B, H, T)
        for h in range(st.H):
            S[h].index_add_(0, x.reshape(-1), diag[:, h].reshape(-1))
        C.index_add_(0, x.reshape(-1), torch.ones(x.numel(), device=dev))
    abar = S / C.clamp(min=1)[None, :]
    add = sum(abar[h][:, None] * st.recall(h) for h in range(st.H))
    st.follow_exc.add_(add)
    st.follow_mask.fill_(1.0)
    st.self_votes.fill_(0.0)
    return abar


def block_median(E: torch.Tensor, row_cls: torch.Tensor, col_cls: torch.Tensor, Kr: int, Kc: int) -> torch.Tensor:
    """(Kr, Kc) medians of E over blocks (class of row token, class of column token)."""
    out = torch.zeros(Kr, Kc, dtype=E.dtype, device=E.device)
    rows = [torch.nonzero(row_cls == k).flatten() for k in range(Kr)]
    cols = [torch.nonzero(col_cls == k).flatten() for k in range(Kc)]
    for a in range(Kr):
        if len(rows[a]) == 0:
            continue
        Ea = E[rows[a]]
        for b in range(Kc):
            if len(cols[b]):
                out[a, b] = Ea[:, cols[b]].median()
    return out


@torch.no_grad()
def recenter(st: Student, rounds: int = 3) -> None:
    """Lossless median polish of the exception tables (in place; run on a CPU student with all masks 1).

    The tables are defined only up to exact shifts: each FOLLOW / RECALL row has a free constant (the softmax
    ignores it), FOLLOW column effects belong in WORD, class-block effects in CLASS_FOLLOW / CLASS_RECALL and
    class-wise diagonal effects in COPY. Adam leaves these in arbitrary places, so "prune = set to 0" would
    reset entries to an arbitrary level. After polishing, zero means "the row's typical value" and pruning
    removes only real deviations. RECALL column effects are moved into WORD only when heads vote on the
    current token too (otherwise vote weights do not sum to 1 and the move is not exact).
    Blocks are (CONTEXT class of the row token, TARGET class of the column token) and COPY's diagonal effects go
    by CONTEXT class, exactly the structure of the class tables, so every move is exact.
    Idea and measurements: design-panel sparsify proposal (logs/compression_experiments.md)."""
    assert bool((st.follow_mask == 1).all()) and bool((st.recall_mask == 1).all()), "recenter a dense student"
    ctx, tgt, Kr, Kc = st.ctx.cpu(), st.tgt.cpu(), st.Kr, st.Kc
    recall_columns = float(st.self_votes) == 1.0
    E = st.follow_exc.detach().clone()
    for _ in range(rounds):
        E -= E.median(1, keepdim=True).values
        k = E.median(0).values
        E -= k[None, :]
        st.word += k
        b = block_median(E, ctx, tgt, Kr, Kc)
        E -= b[ctx][:, tgt]
        st.class_follow += b
    st.follow_exc.copy_(E)
    for h in range(st.H):
        R = st.recall_exc[h].detach().clone()
        for _ in range(rounds):
            R -= R.median(1, keepdim=True).values
            if recall_columns:
                k = R.median(0).values
                R -= k[None, :]
                st.word += k
            b = block_median(R, ctx, tgt, Kr, Kc)
            R -= b[ctx][:, tgt]
            st.class_recall[h] += b
            d = torch.diagonal(R).clone()
            dc = torch.zeros(Kr, dtype=R.dtype)
            for c in range(Kr):
                m = ctx == c
                if m.any():
                    dc[c] = d[m].median()
            R -= torch.diag(dc[ctx])
            st.copy[h] += dc
        st.recall_exc[h].copy_(R)


def prune_global(st: Student, n_total: int, W: torch.Tensor) -> dict:
    """Keep the n_total most important exceptions across FOLLOW and all heads' RECALL tables."""
    i_f, i_r = data_importance(st, W)
    allimp = torch.cat([i_f.flatten(), i_r.flatten()])
    with torch.no_grad():
        if n_total < int((allimp > 0).sum()):
            thr = torch.topk(allimp, n_total).values[-1]
        else:
            thr = torch.tensor(1e-30, device=allimp.device)
        st.follow_mask *= ((i_f >= thr) & (i_f > 0)).float()
        st.recall_mask *= ((i_r >= thr) & (i_r > 0)).float()
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
    if st.H == 0:
        return []
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
        log_every=100, seed=0, tgt: Target | None = None, W=None, tau=0.0, freeze_attention=False,
        freeze=(), train_conf: bool = False):
    torch.manual_seed(seed)
    tgt = tgt or Target(model_id, device)
    W = W if W is not None else torch.as_tensor(train_windows(n_windows))
    st.to(device)
    frozen = set(freeze) | ({"salience", "prefer", "recency"} if freeze_attention else set())
    if not train_conf:  # CONFIDENCE gave nothing on v9 (design-panel probe); keep it out unless asked
        frozen.add("log_conf")
    params = [p for n, p in st.named_parameters() if n not in frozen]
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


SPARSE = ("follow_exc", "recall_exc")


def save_student(st: Student, path: str) -> None:
    """Save a Student compactly: exception tables as (flat index, value) pairs of their kept entries
    (a 60k-entry student is ~1 MB instead of 160 MB); everything else as is."""
    sd = {k: v.detach().cpu() for k, v in st.state_dict().items()}
    out = {"format": "sparse-v1", "n_heads": st.H, "n_classes": st.K}
    for name, mask in (("follow_exc", "follow_mask"), ("recall_exc", "recall_mask")):
        m = sd.pop(mask).flatten() != 0
        w = sd.pop(name).flatten()
        idx = torch.nonzero(m).flatten()
        out[name] = {"shape": tuple(st.state_dict()[name].shape), "idx": idx.to(torch.int64), "val": w[idx]}
    out["dense"] = sd
    torch.save(out, path)


def student_like(sd: dict, n_heads: int) -> Student:
    """An empty Student with the class maps stored in a state dict (cls; ctx and tgt if present)."""
    cls = sd["cls"].numpy()
    ctx = sd["ctx"].numpy() if "ctx" in sd else None
    tgt = sd["tgt"].numpy() if "tgt" in sd else None
    return Student(n_heads, cls, int(sd["prefer"].shape[1]), context_of=ctx, target_of=tgt)


def load_student(path: str, class_of=None) -> Student:
    """Load a Student saved by save_student, or a legacy dense state dict. The class maps are the checkpoint's
    own (class_of is used only by checkpoints without a stored map)."""
    obj = torch.load(path, map_location="cpu")
    if obj.get("format") != "sparse-v1":
        if "cls" not in obj:
            obj = dict(obj, cls=torch.as_tensor(class_of))
        st = student_like(obj, obj["salience"].shape[0])
        st.load_state_dict(obj)
        return st
    sd = dict(obj["dense"])
    st = student_like(sd, obj["n_heads"])
    for name, mask in (("follow_exc", "follow_mask"), ("recall_exc", "recall_mask")):
        shape = obj[name]["shape"]
        n = int(np.prod(shape))
        w = torch.zeros(n)
        m = torch.zeros(n)
        w[obj[name]["idx"]] = obj[name]["val"]
        m[obj[name]["idx"]] = 1.0
        sd[name] = w.view(shape)
        sd[mask] = m.view(shape)
    st.load_state_dict(sd)
    return st
