"""Open-track insight (no numbers leave this script): what does each head write when it reads a word start or a
word piece? Linearised vote u_h(s) = W_E (g_f * centre(W_O^h W_V^h LN1(e_s + mean position))) compared with corpus
statistics of s, by the median over tokens s of the correlation over the vocabulary.

  python -m analysis.investigations.word.ov_probe
"""
from __future__ import annotations

import numpy as np
import torch

from analysis.investigations.word import wordstats as WS
from analysis.mech_eval import program
from harness.common import load_model

V = 2000


def smoothed_pmi(C, p, alpha=44.0):
    tot = C.sum(1, keepdims=True)
    return np.log((C + alpha * p[None, :]) / (tot + alpha)) - np.log(p)[None, :]


def main():
    P = program()
    s = P.STATS
    m = load_model("attn_s0")
    D, H, Dh = 128, 4, 32
    with torch.no_grad():
        E = m.wte.weight.double()
        pos = m.wpe.weight.double().mean(0)
        blk = m.blocks[0]
        x = torch.nn.functional.layer_norm(E + pos, (D,), weight=blk.ln1.weight.double())
        Wv = blk.attn.qkv.weight.double()[2 * D:]
        Wo = blk.attn.out.weight.double()
        gf = m.ln_f.weight.double()
        U = {}
        for h in range(H):
            v = x @ Wv[h * Dh:(h + 1) * Dh].T                       # (V, Dh)
            o = v @ Wo[:, h * Dh:(h + 1) * Dh].T                    # (V, D)
            o = o - o.mean(1, keepdim=True)
            U[h] = ((o * gf) @ E.T).numpy()                          # (V tokens read, V votes)
        xd = torch.nn.functional.layer_norm(E + pos, (D,), weight=None)
        direct = ((xd - xd.mean(1, keepdim=True)) * gf) @ E.T
    p = s.p_uni
    stats = {"pmi1": np.asarray(s.pmi[1]), "pmi2": np.asarray(s.pmi[2]), "pmi3": np.asarray(s.pmi[3])}
    for k in ["W2", "WANY", "WEND", "WPIECE"]:
        stats[k] = smoothed_pmi(np.load(f"{WS.OUT}/{k}.npy").astype(np.float64), p)
    stats["copy"] = np.eye(V) * 10
    w = np.sqrt(p)
    w = w / w.sum()

    def wcorr(a, b):
        a = a - (w * a).sum(-1, keepdims=True)
        b = b - (w * b).sum(-1, keepdims=True)
        return (w * a * b).sum(-1) / np.sqrt((w * a * a).sum(-1) * (w * b * b).sum(-1) + 1e-30)

    onset = np.where((s.shape == 0) & (s.cont > 0.5) & (s.unigram > 20000))[0]
    piece = np.where((s.piece > 0) & (s.unigram > 20000))[0]
    word = np.where((s.shape == 0) & (s.cont < 0.1) & (s.unigram > 20000))[0]
    for label, toks in (("onsets (cont>0.5)", onset), ("word pieces", piece), ("whole words (cont<0.1)", word)):
        print(f"{label}: {len(toks)} tokens; median weighted corr of the head's vote with each statistic")
        for h in range(H):
            row = {k: float(np.median(wcorr(U[h][toks], st[toks]))) for k, st in stats.items()}
            print(f"   head {h}: " + "  ".join(f"{k} {v:+.2f}" for k, v in row.items()))
        row = {k: float(np.median(wcorr(direct.numpy()[toks], st[toks]))) for k, st in stats.items()}
        print("   direct: " + "  ".join(f"{k} {v:+.2f}" for k, v in row.items()))
        # norms
        print("   vote std (weighted) per head:", [round(float(np.median(np.sqrt((w * (U[h][toks] - (w * U[h][toks]).sum(-1, keepdims=True)) ** 2).sum(-1)))), 2) for h in range(H)])


if __name__ == "__main__":
    main()
