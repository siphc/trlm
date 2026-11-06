# Pilot

## LR check (mlp seed 0, 12M tokens)

| peak LR | dev_full loss | tokens | seconds | device |
|---|---|---|---|---|
| 0.0005 | 3.6697 | 12,484,608 | 352 | mps |
| 0.001 | 3.2331 | 12,484,608 | 702 | mps |
| 0.002 | 2.9171 | 12,484,608 | 74 | mps |

Best LR: 0.002

## Budget ladder (mlp seed 0, best LR, full cosine each)

| budget tokens | dev_full loss | rel. to best | seconds | device |
|---|---|---|---|---|
| 12,484,608 | 2.9171 | +30.42% | 74 | mps |
| 49,987,584 | 2.4378 | +8.99% | 187 | mps |
| 199,999,488 | 2.2923 | +2.49% | 829 | mps |
| 527,482,880 | 2.2367 | +0.00% | 3352 | mps |

Chosen budget (smallest within 1% of best): **527,482,880 tokens** at peak LR 0.002
## Notes

- Assumptions: LR chosen at the smallest budget (12.5M) and reused for the ladder, per AGENTS.md section 5.
  The 800M rung exceeds one epoch, so the top rung is exactly one epoch (32,195 steps × 16,384 tokens).
- **The best LR is at the top edge of the grid.** An extra `explore:` run at 4e-3 gave 2.7340 at 12.5M
  (vs 2.9171 for 2e-3, 6% better) but only 2.2813 at 200M (vs 2.2923, **0.48%** better). At real budgets
  the grid's edge costs under 0.5%, so the protocol choice (2e-3) stands. Not used for any decision.
- **The one-epoch run is the best**, and 200M is 2.5% worse, so the rule picks the epoch cap. The model is
  still improving at one epoch. Budget for all 10 models: 527,482,880 tokens, peak LR 2e-3.
- Seconds for the first two LR runs are inflated (machine throttling, see surprises.md); the 50M/200M/epoch
  runs overlapped with other jobs on the same GPU for part of their time.
- Cost of the 10 final runs at ~250–300k tokens/s on MPS: ~30 min per `attn` model, ~35 min per `mlp`
  model, ~5.5 h total.
