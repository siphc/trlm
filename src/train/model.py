"""One-block transformer used for every model in this project.

Architecture (frozen in configs/frozen.yaml):
  tokens -> token embedding + learned absolute position embedding
         -> x + Attn(LN1(x))
         -> x + MLP(LN2(x))          (only in the `mlp` variant)
         -> LN_f -> logits = h @ W_E^T   (tied input/output embedding)

No bias terms anywhere (nn.Linear(bias=False), nn.LayerNorm(bias=False)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2000
    context_length: int = 256
    n_blocks: int = 1
    d_model: int = 128
    n_heads: int = 4
    mlp: bool = False
    d_hidden: int = 512
    init_std: float = 0.02

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out.is_residual_proj = True

    def forward(self, x: torch.Tensor, return_pattern: bool = False):
        B, T, D = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        if return_pattern:
            # Explicit path, used by analysis tools. Mathematically identical to SDPA.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            pattern = scores.softmax(dim=-1)
            z = pattern @ v
        else:
            pattern = None
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2).reshape(B, T, D)
        return self.out(z), pattern


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_hidden, bias=False)
        self.fc_out = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)
        self.fc_out.is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.relu(self.fc_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = Attention(cfg)
        if cfg.mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
            self.mlp = MLP(cfg)
        else:
            self.ln2 = None
            self.mlp = None

    def forward(self, x: torch.Tensor, return_pattern: bool = False):
        a, pattern = self.attn(self.ln1(x), return_pattern=return_pattern)
        x = x + a
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x, pattern


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            if getattr(m, "is_residual_proj", False):
                std = std / math.sqrt(2 * self.cfg.n_blocks)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)

    def forward(self, idx: torch.Tensor, return_patterns: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.context_length
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        patterns = []
        for blk in self.blocks:
            x, p = blk(x, return_pattern=return_patterns)
            patterns.append(p)
        logits = self.ln_f(x) @ self.wte.weight.T
        if return_patterns:
            return logits, patterns
        return logits


def param_groups(model: TinyLM, weight_decay: float):
    """Weight decay on Linear weights only; not on embeddings or LayerNorm."""
    decay, no_decay = [], []
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if isinstance(mod, nn.Linear):
                decay.append(p)
            else:
                no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def count_params(model: nn.Module) -> dict:
    out = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0] if not name.startswith("blocks") else ".".join(name.split(".")[:3])
        out[top] = out.get(top, 0) + p.numel()
    out["total"] = sum(p.numel() for p in model.parameters())
    return out
