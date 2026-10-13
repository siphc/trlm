# Phase 4 report: the 10 final models

Assumptions: training on the laptop's Apple M4 GPU (MPS, fp32), frozen.yaml v2 (human-approved); budget one epoch
(527,482,880 tokens) at peak LR 2e-3 for every model (logs/pilot.md); seeds 0-4 per variant (init and data order).

| model | dev_full loss | wall time |
|---|---|---|
| attn_s0 | 2.5095 | 53 min |
| attn_s1 | 2.5085 | 96 min (resumed after a battery pause) |
| attn_s2 | 2.5404 | 64 min |
| attn_s3 | 2.4913 | 58 min |
| attn_s4 | 2.5073 | 59 min |
| mlp_s0 | 2.2377 | 45 min |
| mlp_s1 | 2.2415 | 49 min |
| mlp_s2 | 2.2434 | 62 min |
| mlp_s3 | 2.2456 | 72 min |
| mlp_s4 | 2.2398 | 47 min |

- Seed spread: attn 2.491-2.540 (sd 0.018), mlp 2.238-2.246 (sd 0.003). The MLP is worth about 0.27 nats of loss.
- Wall times vary with GPU sharing (fits and other jobs ran alongside); a resume bug on MPS was fixed on the way
  (logs/surprises.md).
- Probe samples (2000 windows sampled at temperature 1 from each model, seed 12345) exist for attn_s0-s4; mlp probes
  not yet built.
- Open item (AGENTS.md section 14), coherence: samples from mlp_s0 are locally fluent and grammatical but drift
  semantically within a few sentences and invent words ("Scunny", "osters", "incredi water"), e.g. "Once upon a time,
  there was a bunny who lived in a big house. He lived in a big hunter. One day, he found a long lamp in the grass."
  At 0.49M parameters the model is a strong local-phrase model, not a story model.
