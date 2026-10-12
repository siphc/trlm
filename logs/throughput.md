# Throughput (AGENTS.md section 5)

Two-minute tests, random tokens, batch 64 × 256, AdamW + clip, fp32. Raw rows in `throughput.jsonl`.

| device | variant | tokens/s |
|---|---|---|
| M4 CPU, 4 threads (torch default) | mlp | 136,707 |
| M4 CPU, 4 threads | attn | 173,937 |
| M4 CPU, 10 threads | mlp | 36,565 (efficiency cores hurt; keep the default) |
| M4 GPU (MPS) | mlp | 258,210 |
| M4 GPU (MPS) | attn | 184,302 (run overlapped with another job; the 10M-token sanity run did ~250k incl. eval) |
| Colab GPU | — | **not measured**: needs a human to run `notebooks/train_colab.ipynb` |

Extrapolation at ~250k tokens/s (MPS):

| job | tokens | time |
|---|---|---|
| LR check, 3 × 12.5M | 37.5M | ~3 min |
| Budget ladder 50M + 200M + 527.5M (12.5M reused) | 777.5M | ~52 min |
| 10 final runs at 200M | 2.0B | ~2.2 h |
| 10 final runs at 527.5M (one epoch) | 5.3B | ~5.9 h |

On CPU (~140k–170k) multiply by ~1.6.
