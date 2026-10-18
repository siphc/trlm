"""Study A, "fixed_votes" (logs/plan_attention_law_and_readability.md section 4; pre-registration
logs/prereg_fixed_votes.md): a one-block attention model whose heads vote with fixed corpus tables.

    logits_i = beta_follow * log P1(. | t_i)  +  sum_h beta_h * sum_{j <= i} a^h_ij * T_h(t_j, .)

Only where the heads look is learned: token and position embeddings, one LayerNorm and W_Q, W_K per head (computed
as in TinyLM's attention), plus the scalars beta_follow and beta_h. log P1 (FOLLOW's bigram rule) and the head tables
T_h are counted from the training stories (program/corpus.py, via the analysis cache) and stored once in
data/fixed_votes/tables.npz by build_tables(). They are buffers, not parameters, and are not written into
checkpoints; the checkpoint's config records their sha, which is checked on load.

Head tables (named, distance-independent like a real head's vote). Each is row-centred under unigram weights (free:
it adds the same number to every logit of a position) and divided by its unigram-weighted RMS, so that the betas of
different heads are comparable:
  next   PMI_1(s, v)  what usually follows s
  copy   1[v = s]     repeat s
  topic  topic PMI    v sits 9..24 tokens after s
  skip2  PMI_2(s, v)  what comes two tokens after s
  skip3  PMI_3(s, v)  three tokens after s (the intervention variant)

  .venv/bin/python -m src.train.fixed_votes build      # writes data/fixed_votes/tables.npz (once)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES = os.path.join(REPO, "data", "fixed_votes", "tables.npz")
TABLE_NAMES = ("next", "copy", "topic", "skip2", "skip3")
V = 2000


def build_tables(path: str = TABLES) -> str:
    """Write log P1 ('follow'), the five head tables and the unigram weights to `path`; returns the sha."""
    sys.path.insert(0, REPO)
    from analysis.mech_eval import stats

    s = stats()
    pu = np.asarray(s.p_uni, np.float64)
    pu = pu / pu.sum()
    raw = {"next": s.pmi[1], "copy": np.eye(V), "topic": s.topic, "skip2": s.pmi[2], "skip3": s.pmi[3]}
    out = {"follow": np.asarray(s.log_follow, np.float32), "p_uni": pu.astype(np.float32)}
    rms = {}
    for name in TABLE_NAMES:
        t = np.asarray(raw[name], np.float64)
        t = t - (t @ pu)[:, None]
        rms[name] = float(np.sqrt(pu @ ((t * t) @ pu)))
        out[name] = (t / rms[name]).astype(np.float32)
    sha = tables_sha(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out, sha=sha, rms=json.dumps(rms))
    return sha


def tables_sha(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        if k in ("sha", "rms"):
            continue
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k], np.float32).tobytes())
    return h.hexdigest()[:16]


@dataclass
class FixedVoteConfig:
    tables: str = "next,copy,topic,skip2"   # one table name per head, in head order
    arch: str = "fixed_votes"               # marks the checkpoint for src.train.train.load_model
    vocab_size: int = V
    context_length: int = 256
    d_model: int = 128
    d_head: int = 32
    init_std: float = 0.02
    beta_init: float = 0.1                  # every head's vote strength at initialisation (beta_follow starts at 1)
    tables_sha: str = ""                    # filled in from the tables file; checked on load

    @property
    def names(self) -> list[str]:
        return self.tables.split(",")

    @property
    def n_heads(self) -> int:
        return len(self.names)


class FixedVoteLM(nn.Module):
    def __init__(self, cfg: FixedVoteConfig, tables_path: str = TABLES):
        super().__init__()
        assert all(n in TABLE_NAMES for n in cfg.names), cfg.tables
        z = np.load(tables_path)
        sha = str(z["sha"])
        if cfg.tables_sha and cfg.tables_sha != sha:
            raise ValueError(f"tables file {tables_path} has sha {sha}, the checkpoint expects {cfg.tables_sha}")
        cfg.tables_sha = sha
        self.cfg = cfg
        H, Dh, D = cfg.n_heads, cfg.d_head, cfg.d_model
        self.wte = nn.Embedding(cfg.vocab_size, D)
        self.wpe = nn.Embedding(cfg.context_length, D)
        self.ln = nn.LayerNorm(D, bias=False)
        self.qk = nn.Linear(D, 2 * H * Dh, bias=False)          # q = x @ W[:H*Dh].T, k = x @ W[H*Dh:].T
        self.beta_follow = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.full((H,), cfg.beta_init))
        self.register_buffer("follow", torch.as_tensor(z["follow"]), persistent=False)
        self.register_buffer("tables", torch.as_tensor(np.stack([z[n] for n in cfg.names])), persistent=False)
        for m in (self.wte, self.wpe, self.qk):
            nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)

    def patterns(self, idx: torch.Tensor) -> torch.Tensor:
        """Attention patterns (B, H, T, T), causal softmax of q.k / sqrt(d_head)."""
        B, T = idx.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        x = self.ln(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))[None])
        q, k = self.qk(x).split(H * Dh, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=idx.device).tril(), float("-inf"))
        return sc.softmax(-1)

    def forward(self, idx: torch.Tensor, return_patterns: bool = False, override: dict | None = None):
        """Logits (B, T, V). override = {head: pattern (B, T, T)} replaces those heads' attention (analysis, tests)."""
        a = self.patterns(idx)
        logits = self.beta_follow * self.follow[idx]
        for h in range(self.cfg.n_heads):
            ah = a[:, h] if not override or h not in override else override[h].to(a.dtype)
            logits = logits + self.beta[h] * (ah @ self.tables[h][idx])
        if return_patterns:
            return logits, [a]
        return logits


def load(st: dict, device: str = "cpu", tables_path: str = TABLES) -> FixedVoteLM:
    """A FixedVoteLM from a weights checkpoint dict (src.train.train.save_weights)."""
    model = FixedVoteLM(FixedVoteConfig(**st["model_config"]), tables_path)
    model.load_state_dict(st["model"])
    return model.to(device).float().eval()


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(build_tables())
    else:
        raise SystemExit(__doc__)
