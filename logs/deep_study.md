# Side study: "where a head looks comes from what it writes" in deeper models (running log)

Plan: ~/.claude/plans/stateful-zooming-deer.md (approved 2026-10-14), summarised in logs/decisions.md item 8.

## 2026-10-14 training code and checks

- `n_blocks` added to TrainConfig (default 1; main track unchanged: test_main_track_unchanged reproduces attn_s0's
  dev_full loss). New commands: `python -m src.train.runs deep_lr` and `deep --n_blocks L --seeds ...`.
- src/train/test_sanity_deep.py: 14 tests pass (causality, SDPA = explicit, single-batch overfit to < 0.01 at 2 and 4
  blocks, parameter counts 354,560 / 420,224 / 551,552 and residual-projection init, CPU resume round trip,
  main track unchanged, short runs). Parameter counts in logs/param_counts_deep.json.
- Short runs (25M tokens, peak LR 2e-3; logs/sanity_short_run_deep.json): dev_full loss 1 block 3.041, 2 blocks
  3.137, 4 blocks 3.179 (bigram 3.516, unigram 5.984). At 10M tokens and LR 1e-3, as in test_sanity.py's mlp test,
  the 2-block model was still above the bigram (3.948): attention-only models learn more slowly early on (attn_s0
  itself was at 3.545 after 8M tokens).
- Throughput, MPS fp32, 60 s each: 1 block 325k tok/s, 2 blocks 213k (epoch ~41 min), 4 blocks 124k (~71 min).
  4 blocks is above 110k, so all four confirmatory 4-block seeds are kept. Queue estimate ~9.3 h.

## 2026-10-14 learning-rate check and training queue

- 4 blocks, seed 0, 50M tokens: peak LR 1e-3 gives dev_full 2.742; 2e-3 gives 2.622, with no NaN and no loss spike
  (largest rise within 200 steps after warmup: 0.08 nats). The rule keeps the frozen 2e-3.
- Training queue started (scripts/deep_queue.sh, on the charger, caffeinate): 2L s0, 4L s0, 2L s1-s4, 4L s1-s4.
  scripts/battery_watchdog.sh stops it on battery below 25%; re-running the queue resumes.
- Deep analysis code in analysis/look_write_deep/. Selftests pass on attn_s0 and on the 2L/4L short-run models
  (explicit forward = model; own-pattern substitution exact; float64 gradients = finite differences to < 1e-7;
  SAME = TOTAL for the last block). Classifier reproduces the one-layer LOCAL/RECALL split.
- Calibration (1M-token sample, attn_s0): the DV readout reproduces the one-layer UOV tables (r = 0.96-0.98 at
  d = 1, 2); the exact-gradient readout TOTAL does not (r = 0.76-0.86), because the final LayerNorm's per-position
  Jacobian differs from its typical-scale linearisation. Added LTOTAL (later blocks exact, final LayerNorm
  linearised), which equals DV exactly for one block. Which readout is primary is decided on the one-layer models
  before the deep pre-registration.
- Resource rule revised (not part of the science): with one CPU analysis job (corpus pass) at the lowest priority
  (renice 20), 2-block training runs at 165-168k tok/s, 78% of the measured 213k; the 80% pause threshold would hold
  all analysis back for ~9 h, while continuing costs ~2 h of training. Threshold set to 75% (160k for 2 blocks,
  93k for 4 blocks); analysis jobs run at nice 20.
- Code test, development model attn_s0 only (not held out): local heads' direct-path vote tables have 15-22% of their
  energy in the top-32 subspaces of the corpus statistic C_res (random 32-dim subspaces of its top 128: ~4%), but
  39-67% in the top-32 subspaces of the model's own embedding; C_pmi scores above C_res; the RECALL head (h1) scores
  above the LOCAL heads. Expectation for the held-out heads (to be written into the pre-registration): K1's
  embedding comparison, K3 and K4 likely fail.

## Decision rules fixed before any calibration output (2026-10-14)

Calibration = the deep-study pipeline (pass_bgu + fit_arms_deep) run on the one-layer models attn_s1..s4 with the same
200M-token sample size as the confirmatory runs (MPS, after the training queue). The 20M-token CPU passes now running
are code checks only and set nothing.
- R1, primary readout: LTOTAL, unless TOTAL's median own-arm gain on the 8 confirmatory one-layer heads exceeds
  LTOTAL's by more than 10% (then TOTAL).
- R2, theta for D1: 0.5, unless the median own/oracle gain ratio of those 8 heads (primary readout) is below 0.75;
  then theta = 2/3 x that median, rounded down to a multiple of 0.05.
- R3, D5 references: c1, c2 = medians of the own-arm coefficients of those 8 heads (primary readout).
- R4, sample: 200M tokens of aligned TRAIN windows (default_rng(7) order), fit and eval windows excluded, batch 32.
- The deep pre-registration (logs/prereg_deep_look_write.md) is committed after R1-R3 are applied and after the
  development seeds (2L s0, 4L s0) are classified and self-tested, and before any arm is fitted on seeds 1-4.
