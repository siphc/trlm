"""Open-track analysis tools (AGENTS.md phase 5). Insight may come from the weights; numbers copied
from the weights may not go into program/.

  python -m analysis.tools ablate     --model attn_s0      # per-head ablation loss / KL deltas
  python -m analysis.tools attn       --model attn_s0      # attention pattern summaries per head
  python -m analysis.tools worst      --model attn_s0      # worst-KL tokens vs a reference program
  python -m analysis.tools sample     --model attn_s0      # decoded samples at temperature 1

Library use: `from analysis.tools import Model; m = Model("attn_s0"); r = m.run(tokens)` gives the
residual-stream components, attention patterns, LN scales and per-component logits for one window.
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

from harness.common import eval_windows, load_model, model_logprobs  # noqa: E402
from src.train.data import load_tokenizer  # noqa: E402


def kl_rows(p_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    p = np.exp(p_log)
    return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)


class Model:
    """Weights as numpy arrays plus a decomposed forward pass for one block."""

    def __init__(self, model_id: str):
        self.torch_model = load_model(model_id)
        sd = {k: v.numpy().astype(np.float64) for k, v in self.torch_model.state_dict().items()}
        self.cfg = self.torch_model.cfg
        self.WE = sd["wte.weight"]                          # (V, D)
        self.WP = sd["wpe.weight"]                          # (T, D)
        self.g1 = sd["blocks.0.ln1.weight"]
        self.Wqkv = sd["blocks.0.attn.qkv.weight"]          # (3D, D), y = x W^T
        self.WO = sd["blocks.0.attn.out.weight"]            # (D, D)
        self.gf = sd["ln_f.weight"]
        self.mlp = self.cfg.mlp
        if self.mlp:
            self.g2 = sd["blocks.0.ln2.weight"]
            self.Win = sd["blocks.0.mlp.fc_in.weight"]      # (H, D)
            self.Wout = sd["blocks.0.mlp.fc_out.weight"]    # (D, H)
        self.tok = load_tokenizer()

    @staticmethod
    def ln(x: np.ndarray, g: np.ndarray, eps: float = 1e-5):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        sd = np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps)
        return xc / sd * g, sd[..., 0]

    def run(self, tokens) -> dict:
        t = np.asarray(tokens, dtype=int)
        T, D, H = len(t), self.cfg.d_model, self.cfg.n_heads
        Dh = D // H
        emb, pos = self.WE[t], self.WP[:T]
        x0 = emb + pos
        h1, sd1 = self.ln(x0, self.g1)
        qkv = h1 @ self.Wqkv.T
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        heads_out, patterns = [], []
        mask = np.tril(np.ones((T, T), dtype=bool))
        for h in range(H):
            sl = slice(h * Dh, (h + 1) * Dh)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(Dh)
            s = np.where(mask, s, -np.inf)
            a = np.exp(s - s.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            patterns.append(a)
            heads_out.append((a @ v[:, sl]) @ self.WO[:, sl].T)
        x1 = x0 + sum(heads_out)
        comps = {"emb": emb, "pos": pos, **{f"head{h}": heads_out[h] for h in range(H)}}
        x = x1
        if self.mlp:
            h2, _ = self.ln(x1, self.g2)
            act = np.maximum(h2 @ self.Win.T, 0)
            mlp_out = act @ self.Wout.T
            comps["mlp"] = mlp_out
            x = x1 + mlp_out
        hf, sdf = self.ln(x, self.gf)
        logits = hf @ self.WE.T
        # Per-component logits: final LN is linear given its scale sdf, so
        # logits = sum_c ((c - mean(c)) * gf / sdf) @ WE^T exactly.
        comp_logits = {name: ((c - c.mean(-1, keepdims=True)) * self.gf / sdf[:, None]) @ self.WE.T
                       for name, c in comps.items()}
        return {"tokens": t, "logits": logits, "patterns": np.stack(patterns), "ln1_scale": sd1,
                "lnf_scale": sdf, "components": comps, "comp_logits": comp_logits,
                "mlp_act": act if self.mlp else None}

    def decode(self, ids) -> list[str]:
        return [self.tok.id_to_token(int(i)) for i in ids]


def cmd_ablate(model_id: str, split: str, n: int) -> dict:
    m = load_model(model_id)
    w = eval_windows(split)[:n]
    y = w[:, 1:].astype(int)
    base = model_logprobs(m, w)
    r = np.arange(256)

    def ce(lp):
        return float(-np.take_along_axis(lp, y[..., None], -1).mean())

    out = {"model": model_id, "split": split, "n_windows": n, "ce_model": ce(base), "heads": {}}
    H = m.cfg.n_heads
    for h in list(range(H)) + ["all"]:
        heads = list(range(H)) if h == "all" else [h]
        lp = model_logprobs(m, w, ablate_heads=heads)
        out["heads"][str(h)] = {"ce": ce(lp), "d_ce": ce(lp) - ce(base),
                                "kl_from_full": float(kl_rows(base, lp).mean())}
    print(json.dumps(out, indent=2))
    return out


def cmd_attn(model_id: str, split: str, n: int) -> None:
    M = Model(model_id)
    w = eval_windows(split)[:n]
    H = M.cfg.n_heads
    stats = {h: {"self": 0.0, "prev": 0.0, "bos_like_eos": 0.0, "mean_dist": 0.0, "entropy": 0.0} for h in range(H)}
    for win in w:
        r = M.run(win[:256])
        T = 256
        dist = np.arange(T)[:, None] - np.arange(T)[None, :]
        is_eos = (win[:256] == 0)[None, :]
        for h in range(H):
            a = r["patterns"][h]
            stats[h]["self"] += np.trace(a) / T
            stats[h]["prev"] += np.trace(a, offset=-1) / (T - 1)
            stats[h]["bos_like_eos"] += float((a * is_eos).sum(1).mean())
            stats[h]["mean_dist"] += float((a * np.maximum(dist, 0)).sum(1).mean())
            stats[h]["entropy"] += float(-(a * np.log(a + 1e-30)).sum(1).mean())
    for h in range(H):
        for k in stats[h]:
            stats[h][k] /= len(w)
    print(json.dumps({"model": model_id, "heads": stats}, indent=2))


def cmd_sample(model_id: str, n: int, temp: float = 1.0, seed: int = 0) -> None:
    m = load_model(model_id)
    tok = load_tokenizer()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        x = torch.zeros((n, 1), dtype=torch.long)
        for _ in range(255):
            p = torch.softmax(m(x)[:, -1] / temp, -1)
            x = torch.cat([x, torch.multinomial(p, 1, generator=g)], 1)
    for row in x.numpy():
        text = tok.decode(row[1:].tolist())
        print(text.split("<|eos|>")[0].strip()[:1200])
        print("-" * 80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ablate", "attn", "sample"])
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--split", default="dev_fast")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    if args.cmd == "ablate":
        cmd_ablate(args.model, args.split, args.n)
    elif args.cmd == "attn":
        cmd_attn(args.model, args.split, min(args.n, 50))
    elif args.cmd == "sample":
        cmd_sample(args.model, min(args.n, 10))


if __name__ == "__main__":
    main()
