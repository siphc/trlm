# Side study: can a language model be trained to be readable? (running log)

Decision: logs/decisions.md item 9. Configuration: configs/side_readable.yaml. Pre-registration:
logs/prereg_readable_training.md (DRAFT committed 2026-10-17, before any study output).

## Plan (approved 2026-10-17)

Train new 1-block attention-only models with an extra loss that pulls them toward the readable program v1.3, then
check with tools that were not trained on whether the agreement is genuine or Goodharted (Hubinger's
"robust-to-training transparency", footnote 10 of "A transparency and interpretability tech tree", at our scale).

| phase | content | human gate |
|---|---|---|
| 0 | decision item 9, config, DRAFT pre-registration | G0: loss design and draft |
| 1 | training hooks (src/train/readable.py), sanity tests, target pool, vote targets, program caches | G1: training code before long runs |
| 2 | baseline measurements on attn_s0..s4 and their refits; sigma for R4 | |
| 3 | development grid (seed 0, 100M tokens) and seed-0 one-epoch runs; rules R1-R7; FINAL pre-registration | G2: dose-response and lambda |
| 4 | confirmatory runs: arms O and I, seeds 1-4, one epoch (overnight, charger) | |
| 5 | measurement suite on the new models | |
| 6 | verdicts, results (logs/readable_training_results.md), dead ends, surprises | G3 |

Compute: laptop MPS only (about 9 h of training) and about 30 CPU-hours of measurement at nice 20.

## 2026-10-17 phase 0

- Decision item 9, configs/side_readable.yaml and the DRAFT pre-registration written. The study's own window seeds
  (9001 pool, 9100 TRAIN evaluation, 9125 attention patterns) are unused elsewhere in the repository; every
  train_windows (n, seed) pair found in the repository's code is excluded from the pool.
- Waiting for gate G0.
