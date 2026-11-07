"""Readable dump of a ladder model (logs/ladder/design.md, Tools): every weight table at the bottom rungs, plus the
first measurements an account needs.

    .venv/bin/python -m analysis.ladder.dump --rung L1 --arm ln --seed 0 [--out logs/ladder/dumps]

Writes <out>/<rung>_<arm>_s<seed>.md:
- the embedding of every symbol (sorted by its first principal direction) and the position embedding summary;
- LayerNorm gains; the QK and OV matrices per head in the embedding basis;
- how much each site matters: KL(model || model with the site resample-ablated) for the direct path and each head;
- attention by distance (mean weight on the key d positions back) per head;
- the direct path alone: KL(model || LN_f(x0) E^T).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from analysis.ladder.common import REPO, kl_pos, load, model_logprobs, windows
from analysis.ladder.scrub import finish, parts, scrubbed
from src.train import data_char


def symbol_names() -> list[str]:
    a = data_char.load_alphabet()
    return ["EOS", "UNK"] + [repr(c)[1:-1] if c not in " " else "␣" for c in a["chars"]]


def fmt(v, p=2) -> str:
    return " ".join(f"{x:+.{p}f}" for x in np.ravel(v))


def attention(model, X, h):
    import math

    import torch

    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.int64)
        n, T = x.shape
        blk = model.blocks[0]
        d, H, dh = model.cfg.d_model, model.cfg.n_heads, model.cfg.d_head
        x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
        q, k, _ = blk.attn.qkv(blk.ln1(x0)).split(d, dim=-1)
        q = q.view(n, T, H, dh)[:, :, h]
        k = k.view(n, T, H, dh)[:, :, h]
        s = q @ k.transpose(1, 2) / math.sqrt(dh)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        return s.softmax(-1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "ladder", "dumps"))
    a = ap.parse_args()
    m = load(a.rung, a.arm, a.seed).double()
    W = {k: v.numpy() for k, v in m.state_dict().items()}
    names = symbol_names()
    E, P = W["wte.weight"], W["wpe.weight"]
    d, H, dh = m.cfg.d_model, m.cfg.n_heads, m.cfg.d_head
    L = [f"# {a.rung} {a.arm} seed {a.seed}: weight dump", "",
         f"d_model {d}, heads {H} x {dh}, context {m.cfg.context_length}, vocab {m.cfg.vocab_size}, "
         f"params {sum(v.size for v in W.values())}", ""]

    Wn = windows(a.rung, "dev_full")
    X = Wn[:, :-1].astype(np.int64)
    lp = model_logprobs(m, Wn)
    tgt = Wn[:, 1:].astype(np.int64)
    L += [f"dev_full loss {-np.take_along_axis(lp, tgt[..., None], -1).mean():.4f} nats/symbol", ""]

    # embeddings
    Ec = E - E.mean(0)
    u = np.linalg.svd(Ec, full_matrices=False)
    L += ["## Token embedding (rows sorted by the first principal direction)", "",
          f"singular values {fmt(u[1])}", "", "| symbol | train count | embedding |", "|---|---|---|"]
    counts = [0, data_char.load_alphabet()["unk_count"]] + data_char.load_alphabet()["counts"]
    for i in np.argsort(Ec @ u[2][0]):
        L.append(f"| `{names[i]}` | {counts[i]:,} | {fmt(E[i])} |")
    L += ["", "## Position embedding", "",
          f"mean {fmt(P.mean(0))}; std over positions {fmt(P.std(0))}; norm of the mean token embedding "
          f"{np.linalg.norm(E.mean(0)):.2f}, mean token-embedding std {E.std(0).mean():.2f}", "",
          "| position | embedding |", "|---|---|"]
    for t in sorted(set([0, 1, 2, 3, 4, 8, 16, 32, P.shape[0] - 1])):
        L.append(f"| {t} | {fmt(P[t])} |")

    # LN and attention matrices
    L += ["", "## LayerNorm gains", ""]
    for k in ("blocks.0.ln1.weight", "ln_f.weight"):
        L.append(f"- {k}: {fmt(W[k]) if k in W else 'none (noln arm)'}")
    Wq, Wk, Wv = np.split(W["blocks.0.attn.qkv.weight"], 3, 0)
    Wo = W["blocks.0.attn.out.weight"]
    g1 = W.get("blocks.0.ln1.weight", np.ones(d))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        QK = (Wq[sl] * g1).T @ (Wk[sl] * g1) / np.sqrt(dh)
        OV = Wo[:, sl] @ (Wv[sl] * g1)
        L += ["", f"## Head {h}", "", "QK (d x d, query rows, key columns; ln1 gains and 1/sqrt(dh) folded in, so it "
              "acts on the normalised embeddings):", "```"]
        L += [fmt(r) for r in QK] + ["```", "OV (d x d, output rows, input columns; ln1 gains folded in):", "```"]
        L += [fmt(r) for r in OV] + ["```"]
        A = attention(m, X, h)
        T = X.shape[1]
        by_d = [np.mean([A[:, t, t - dist].mean() for t in range(dist, T)]) for dist in range(min(T, 16))]
        L += [f"mean attention on the key `dist` back (dist 0 = self): {fmt(by_d, 3)}"]

    # site importance
    x0, z = parts(m, X)
    rng = np.random.default_rng(0)
    L += ["", "## How much each site matters (dev_full)", "", "| site | KL(model || site resample-ablated) |", "|---|---|"]
    for s in ["direct"] + [f"head{h}" for h in range(H)]:
        L.append(f"| {s} | {kl_pos(lp, scrubbed(m, x0, z, {s: None}, rng)).mean():.4f} |")
    zero = np.zeros_like(z)
    L.append(f"| all heads zeroed | {kl_pos(lp, finish(m, x0, zero)).mean():.4f} |")
    L.append("")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.rung}_{a.arm}_s{a.seed}.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(path)


if __name__ == "__main__":
    main()
