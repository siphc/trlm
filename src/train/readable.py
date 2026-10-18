"""Readable training (side study, logs/readable_training.md; configs/side_readable.yaml): an auxiliary loss that pulls
a model toward the readable program v1.3, computed on a few windows per step from a fixed, precomputed target pool
(analysis/readable_training/build_pool.py).

  Arm O (output level):   lambda_out * KL(program || model), over {the program's top-k tokens, the rest}
  Arm I (internal level): lambda_att * KL(rule || model attention), heads 0, 1, 2 against the rules LOCAL-0, RECALL,
                          LOCAL-2, over {the rule's top-k keys, the rest}
                        + lambda_vote * weighted (1 - row correlation) between each of those heads' vote (OV) table
                          and its corpus vote table (analysis/readable_training/vote_targets.py)
Head 3 and the direct path are unconstrained.

Nothing here touches the global torch or NumPy random state, so a model trained with lambda = 0 is bitwise the model
trained without the hook (src/train/test_sanity_readable.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

T = 256
HEADS = (0, 1, 2)                       # model heads constrained by arm I (rules LOCAL-0, RECALL, LOCAL-2)
SAMPLER_TAG = 9                         # the pool sampler's stream: default_rng([seed, SAMPLER_TAG, epoch])
N_METRIC = 64                           # windows for the online metrics (held-out pool, and pool-training slice)


class Pool:
    """The precomputed targets (memory-mapped). `limit` keeps only the first `limit` training windows."""

    def __init__(self, path: str, limit: int = 0):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        assert self.meta.get("complete"), f"pool at {path} is not complete"
        load = lambda n: np.load(os.path.join(path, f"{n}.npy"), mmap_mode="r")   # noqa: E731
        self.tokens = load("tokens")
        self.n_train = self.meta["n_train"] if not limit else min(limit, self.meta["n_train"])
        self.n_heldout = self.meta["n_heldout"]
        self.out_ids, self.out_lp, self.out_tail = load("out_ids"), load("out_lp"), load("out_tail")
        self.att = [(load(f"att{h}_keys"), load(f"att{h}_p"), load(f"att{h}_tail")) for h in range(3)]
        self.heldout_lp, self.heldout_att = load("heldout_lp"), load("heldout_att")
        self.sha = f"{self.meta['indices_sha']}:{self.meta['program_sha']}:{self.n_train}"

    def train_batch(self, idx: np.ndarray, device, attention: bool):
        idx = np.sort(idx)                                   # faster memmap reads; order within a batch is irrelevant
        t = lambda a, dt: torch.as_tensor(np.asarray(a[idx]), device=device).to(dt)   # noqa: E731
        b = {"x": torch.as_tensor(np.asarray(self.tokens[idx, :T], np.int64), device=device)}
        if attention:
            b["att"] = [(t(k, torch.long), t(p, torch.float32), t(tl, torch.float32)) for k, p, tl in self.att]
        else:
            b["ids"], b["lp"], b["tail"] = t(self.out_ids, torch.long), t(self.out_lp, torch.float32), \
                t(self.out_tail, torch.float32)
        return b


class PoolSampler:
    """Auxiliary windows for step s: positions [s*B, (s+1)*B) of the concatenated per-epoch permutations
    default_rng([seed, SAMPLER_TAG, epoch]).permutation(n), like src/train/data.py BatchSampler. A pure function of
    (seed, step), so resuming needs no extra state."""

    def __init__(self, n: int, seed: int, batch_windows: int):
        self.n, self.seed, self.B = n, seed, batch_windows
        self._cache: dict[int, np.ndarray] = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            self._cache = {epoch: np.random.default_rng([self.seed, SAMPLER_TAG, epoch]).permutation(self.n)}
        return self._cache[epoch]

    def indices(self, step: int) -> np.ndarray:
        out, pos = [], step * self.B
        while len(out) < self.B:
            e, off = divmod(pos, self.n)
            take = min(self.B - len(out), self.n - off)
            out.extend(self._perm(e)[off: off + take].tolist())
            pos += take
        return np.asarray(out, dtype=np.int64)


def topk_forward_kl(logits, ids, lp, tail):
    """KL(p || q) over the partition {the k stored tokens, the rest}, averaged over positions. p: the program (top-k
    log-probs lp and tail mass, renormalised), q = softmax(logits)."""
    lq = F.log_softmax(logits if logits.dtype == torch.float64 else logits.float(), -1)
    p = lp.to(lq.dtype).exp()
    tail = tail.to(lq.dtype)
    s = p.sum(-1) + tail
    p, tail = p / s[..., None], tail / s
    lq_top = lq.gather(-1, ids)
    lq_rest = lq.scatter(-1, ids, float("-inf")).logsumexp(-1)
    kl = (p * (p.clamp_min(1e-30).log() - lq_top)).sum(-1)
    kl = kl + torch.where(tail > 0, tail * (tail.clamp_min(1e-30).log() - lq_rest), torch.zeros_like(tail))
    return kl.mean()


def attention_topk_kl(pattern, keys, p, tail):
    """KL(rule || model) over {the rule's k stored keys, the rest} for one head, query rows i >= 1, averaged.
    pattern (B, T, T); keys (B, T, k); p, tail: the rule's stored probabilities (padding entries have p = 0)."""
    valid = p > 0
    p = torch.where(valid, p, torch.zeros_like(p))
    s = p.sum(-1) + tail
    p, tail = p / s[..., None], tail / s
    m = pattern.gather(-1, keys)
    m_valid = torch.where(valid, m, torch.zeros_like(m))
    kl = torch.where(valid, p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log()), torch.zeros_like(p)).sum(-1)
    m_rest = (1.0 - m_valid.sum(-1)).clamp_min(1e-30)
    kl = kl + torch.where(tail > 0, tail * (tail.clamp_min(1e-30).log() - m_rest.log()), torch.zeros_like(tail))
    return kl[:, 1:].mean()


def ov_tables_torch(model, heads=HEADS):
    """Each head's vote table OV_h(s, v) (direct path only, scale-free): embed s at the mean position, LN1, W_V^h,
    W_O^h, centre over d_model, final LayerNorm gain, tied unembedding. (len(heads), V, V); differentiable. Equal to
    analysis/look_write/common.ov_tables(weights, scale=1)."""
    blk = model.blocks[0]
    D, Dh = model.cfg.d_model, model.cfg.d_head
    pbar = model.wpe.weight[1:T].mean(0)
    X = blk.ln1(model.wte.weight + pbar)
    Wv = blk.attn.qkv.weight[2 * D: 3 * D]
    Wo = blk.attn.out.weight
    out = []
    for h in heads:
        o = (X @ Wv[h * Dh:(h + 1) * Dh].T) @ Wo[:, h * Dh:(h + 1) * Dh].T
        o = o - o.mean(1, keepdim=True)
        out.append((o * model.ln_f.weight) @ model.wte.weight.T)
    return torch.stack(out)


def row_corr(ov, target, colw):
    """rho(s) = colw-weighted correlation over v between ov(s, .) and target(s, .), target rows already standardised
    (sum_v colw t = 0, sum_v colw t^2 = 1)."""
    o = ov - (ov @ colw)[:, None]
    o = o / ((o * o) @ colw).clamp_min(1e-12).sqrt()[:, None]
    return (o * target) @ colw


class ReadableAux:
    def __init__(self, cfg, device):
        assert cfg.readable_arm in ("O", "I"), cfg.readable_arm
        self.arm, self.device = cfg.readable_arm, device
        self.pool = Pool(cfg.readable_pool, cfg.readable_pool_limit)
        self.sampler = PoolSampler(self.pool.n_train, cfg.seed, cfg.aux_windows)
        self.lam = {"out": cfg.lambda_out, "att": cfg.lambda_att, "vote": cfg.lambda_vote}
        self.ramp = int(round(cfg.readable_ramp * cfg.total_steps))
        tg = np.load(cfg.readable_targets)
        self.targets = torch.as_tensor(tg["targets"], dtype=torch.float32, device=device)     # (3, V, V)
        self.colw = torch.as_tensor(tg["colw"], dtype=torch.float32, device=device)
        self.roww = torch.as_tensor(tg["roww"], dtype=torch.float32, device=device)           # (3, V), rows sum to 1
        self.targets_sha = str(tg["sha"])
        self._metric_train = np.random.default_rng([SAMPLER_TAG, 1]).permutation(self.pool.n_train)[:N_METRIC]

    def summary(self) -> dict:
        return {"arm": self.arm, "lambda": self.lam, "ramp_steps": self.ramp, "pool_sha": self.pool.sha,
                "pool_train_windows": self.pool.n_train, "targets_sha": self.targets_sha}

    def check_or_write(self, out_dir: str) -> None:
        p = os.path.join(out_dir, "readable.json")
        s = self.summary()
        if os.path.exists(p):
            old = json.load(open(p))
            assert old == s, f"readable settings differ from the run being resumed: {old} vs {s}"
        else:
            with open(p, "w") as f:
                json.dump(s, f, indent=1)

    def scale(self, step: int) -> float:
        return 1.0 if self.ramp <= 0 else min(1.0, (step + 1) / self.ramp)

    def vote_loss(self, model):
        ov = ov_tables_torch(model)
        rho = torch.stack([row_corr(ov[i], self.targets[i], self.colw) for i in range(len(HEADS))])
        return (self.roww * (1 - rho)).sum(1).mean(), rho

    def loss(self, model, step: int):
        """The weighted auxiliary loss for this step and a dict of detached parts for logging."""
        b = self.pool.train_batch(self.sampler.indices(step), self.device, attention=(self.arm == "I"))
        k = self.scale(step)
        if self.arm == "O":
            l_out = topk_forward_kl(model(b["x"]), b["ids"], b["lp"], b["tail"])
            return k * self.lam["out"] * l_out, {"aux_out": l_out.detach()}
        _, pats = model(b["x"], return_patterns=True)
        l_att = torch.stack([attention_topk_kl(pats[0][:, h], *b["att"][i]) for i, h in enumerate(HEADS)]).mean()
        l_vote, _ = self.vote_loss(model)
        return k * (self.lam["att"] * l_att + self.lam["vote"] * l_vote), \
            {"aux_att": l_att.detach(), "aux_vote": l_vote.detach()}

    @torch.no_grad()
    def metrics(self, model) -> dict:
        """Both arms' quantities, whatever arm is trained: exact KLs to the program and exact pattern KLs to the rules
        on N_METRIC held-out pool windows; the coarsened trained losses on those and on N_METRIC pool-training windows
        (memorisation gap); vote correlations (row-weighted and uniform)."""
        dev = self.device
        res = {"kl_prog_model": 0.0, "kl_model_prog": 0.0, "att_kl_rule_model": 0.0, "att_kl_model_rule": 0.0,
               "heldout_out_coarse": 0.0, "heldout_att_coarse": 0.0}
        k_out = self.pool.out_ids.shape[-1]
        ks = [a[0].shape[-1] for a in self.pool.att]
        nb = 0
        for c in range(0, N_METRIC, 16):
            r = np.arange(c, c + 16)
            x = torch.as_tensor(np.asarray(self.pool.tokens[self.pool.meta["n_train"] + r, :T], np.int64), device=dev)
            lp = torch.as_tensor(np.asarray(self.pool.heldout_lp[r]), device=dev).float()
            lp = lp - lp.logsumexp(-1, keepdim=True)
            rule = torch.as_tensor(np.asarray(self.pool.heldout_att[r]), device=dev).float()
            rule = rule / rule.sum(-1, keepdim=True).clamp_min(1e-30)
            logits, pats = model(x, return_patterns=True)
            lq = F.log_softmax(logits.float(), -1)
            res["kl_prog_model"] += float((lp.exp() * (lp - lq)).sum(-1).mean())
            res["kl_model_prog"] += float((lq.exp() * (lq - lp)).sum(-1).mean())
            topv, topi = lp.topk(k_out, -1)
            res["heldout_out_coarse"] += float(topk_forward_kl(logits, topi, topv,
                                                               (1 - topv.exp().sum(-1)).clamp_min(0)))
            pat = pats[0]
            for i, h in enumerate(HEADS):
                m, a = pat[:, h, 1:], rule[:, i, 1:]
                lm, la = m.clamp_min(1e-30).log(), a.clamp_min(1e-30).log()
                res["att_kl_rule_model"] += float((a * (la - lm)).sum(-1).mean()) / len(HEADS)
                res["att_kl_model_rule"] += float((m * (lm - la)).sum(-1).mean()) / len(HEADS)
                pv, pk = rule[:, i].topk(ks[i], -1)
                res["heldout_att_coarse"] += float(attention_topk_kl(pat[:, h], pk, pv,
                                                                     (1 - pv.sum(-1)).clamp_min(0))) / len(HEADS)
            nb += 1
        res = {k: v / nb for k, v in res.items()}
        # the same coarsened losses on pool-training windows (memorisation gap = held-out - training)
        tr_out, tr_att = 0.0, 0.0
        for c in range(0, N_METRIC, 16):
            idx = self._metric_train[c:c + 16]
            bo = self.pool.train_batch(idx, dev, attention=False)
            ba = self.pool.train_batch(idx, dev, attention=True)
            logits, pats = model(bo["x"], return_patterns=True)
            tr_out += float(topk_forward_kl(logits, bo["ids"], bo["lp"], bo["tail"]))
            tr_att += float(torch.stack([attention_topk_kl(pats[0][:, h], *ba["att"][i])
                                         for i, h in enumerate(HEADS)]).mean())
        res["train_out_coarse"], res["train_att_coarse"] = tr_out / nb, tr_att / nb
        _, rho = self.vote_loss(model)
        for i, h in enumerate(HEADS):
            res[f"vote_rho_weighted_h{h}"] = float((self.roww[i] * rho[i]).sum())
            res[f"vote_rho_uniform_h{h}"] = float(rho[i].mean())
        return {f"m_{k}": v for k, v in res.items()}
