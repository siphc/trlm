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

## 2026-10-18 phase 1 (code, tests, pool, targets)

- **Training code:** src/train/readable.py (pool reader, PoolSampler, the two losses, differentiable vote tables,
  ReadableAux); hooks in src/train/train.py (new TrainConfig fields, default "" = unchanged) and src/train/runs.py
  (`readable_dev`, `readable`). Tests: src/train/test_sanity_readable.py.
- **Selftest 8 (cache):** program outputs from analysis/out/program_stats equal a fresh harness `setup(corpus)` on 3
  pool windows (max |difference| 0.0).
- **Rule R1:** coverage on 256 pool windows: program top-k 0.892 (64), 0.939 (128), 0.973 (256); LOCAL-0 and LOCAL-2
  attention 0.97 at 16 keys; RECALL 0.917 at 64, 0.993 at 128. So k_out 256, k = 16 / 128 / 16. **Deviation:** that
  needs 13.5 GB, above R1's own 10 GB cap; the rule does not say which condition wins. Resolved in favour of coverage
  before the pool was built (the cap was a disk-safety guard; 51 GB free). Recorded in rules.json; R7 (doubling the
  pool) would need about 27 GB.
- **Vote targets** (data/readable/vote_targets.npz): rank-32 energy share (the ceiling for a rank-32 vote table) 0.87
  (LOCAL-0), 0.74 (RECALL), 0.83 (LOCAL-2). Distance profiles: LOCAL-0 puts 27% of its earlier-token attention on d = 1, LOCAL-2 31%.
- **Role assignment** (models.py, cheap: v1.3 rules without refitting): attn_s0 maps to itself; attn_s1 gives LOCAL-0 =
  head 3, RECALL = head 0, LOCAL-2 = head 2, the same as the slow fit_assignment. Aligned checkpoints reproduce the
  logits (max |difference| 1.5e-5).
- **Pool** built: 32,768 + 512 windows, 13 GB, 17 min with 8 workers. Verify: probability mass 1 +- 0.0005 (fp16),
  spot recomputation within 0.004 (fp16 rounding), all finite.
- **Rule R2:** c = 0.836 (on attn_s0, held-out pool: L_att 0.559, L_vote 0.668). attn_s0's vote rows correlate with
  the targets at 0.46 (LOCAL-0), 0.25 (RECALL), 0.29 (LOCAL-2), row-weighted.
- **Rule R6:** per step on MPS with 16 auxiliary windows: baseline 0.058 s, arm O 0.110 s (1.9x), arm I 0.171 s (2.9x);
  both above 1.6x, so 8 auxiliary windows. Training will take about 13 h of MPS instead of the planned 9 h.
- **Tests:** src/train/test_sanity_readable.py 10 of 10 pass (lambda = 0 bitwise identical for both arms; resume bitwise
  for both arms; sampler; KL references and float64 gradient check; head 3 receives no gradient from the attention
  loss; differentiable vote tables equal the analysis code; tiny-pool overfit on the real pool for both arms), and the
  main suite (src/train/test_sanity.py) and test_main_track_unchanged still pass (27 tests in total).
- **Evaluation sets** (analysis/readable_training/evalsets.py; data/readable/cache, 1.8 GB): dev_fast, train_eval, the
  five probe sets, 200 program-sampled windows, sentence- and word-shuffled dev windows, with program v1.3's outputs.
- **Suite smoke test** on attn_s1 (fidelity, internals, lock): the cached dev_fast KL reproduces the harness's logged
  value (0.547115 vs 0.547114). Baseline values for attn_s1: pattern KL to the rules 0.80, vote correlation (uniform)
  0.31, lock 0.48. The generalised tables_vs_attention reproduces attn_s0's arm D (0.1152).
- **Waiting for gate G1** (training code review before long runs). Note: the laptop was on battery during these checks;
  training needs the charger.
