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

## 2026-10-14 development seed attn2L_s0

- dev_full loss 2.2751 (1-block attention-only models: 2.491-2.540; 1-block mlp models: 2.238-2.246).
- Selftests pass (explicit forward = model to 5e-5; float64 gradients = finite differences to < 3e-8).
- Classes: 6 confirmatory heads (L0H0, L0H2, L0H3 in F1; L1H0, L1H2, L1H3 in F2), 2 far-looking (L0H1 20.6 tokens,
  L1H1 44.3). No PREV head (largest previous-token attention 0.30) and no INDUCTION head (largest score 0.03 on repeated
  random sequences).
- Note: a training gap of ~2 h (machine asleep, lid closed) between steps 13000 and 14000; training itself unaffected.

## 2026-10-14 development seed attn4L_s0

- dev_full loss 2.1273 (attn2L_s0 2.2751; 1-block attn 2.491-2.540).
- Selftests pass (forward = model to 6e-5; gradients = finite differences to < 1e-8).
- Classes: 7 confirmatory heads: L0H0, L0H3 (F1); L1H0, L1H1, L2H0, L2H2, L2H3 (F3); none in the last layer (all four
  layer-3 heads look far: mean distances 12.7-35.2). Far-looking: L0H1, L0H2, L1H2, L1H3, L2H1, L3H0-3. No PREV head
  (max previous-token attention 0.26) and no INDUCTION head (max 0.01).
- No in-context copying of repeated random sequences in any model (loss on the second copy vs the first: 1 block
  15.65 vs 16.39, 2 blocks 13.43 vs 14.81, 4 blocks 15.27 vs 15.25 nats; an induction circuit would bring the second
  copy to a few nats). These TinyStories attention-only models form no induction heads.
- Consequence for the families: F2 (last layer) will come mostly from the 2-block models.

## Status and how to resume (2026-10-15, laptop about to sleep)

- Trained: attn2L_s0 (2.2751), attn4L_s0 (2.1273), attn2L_s1 (2.2665); attn2L_s2 training. Still to train: attn2L_s3,
  attn2L_s4, attn4L_s1..s4 (~6-7 h on the charger).
- Done in the analysis: addendum A, B, C (logs/look_follows_write_addendum_results.md); dev seeds classified and
  self-tested; attn2L_s1 classified and self-tested (5 confirmatory heads); 20M-token code-check passes on attn_s1..s3.
- Running on the CPU: the code-check pass on attn_s4, then the one-layer arm fits (ltotal, total), then addendum D (echo
  trajectory, 20 checkpoint passes at 5M tokens).
- Sleep needs nothing: processes pause and continue on wake. After a shutdown or crash: `sh scripts/resume_deep_study.sh`
  (idempotent: starts the training queue, the battery watchdog and the CPU queue if they are not running; finished
  models, passes and arm files are skipped).
- Next steps once training is done (MPS free):
  1. 200M-token passes on attn_s1..s4 (MPS), fit_arms_deep with both readouts, then apply R1-R3 (above) to fix the
     primary readout, theta and the D5 references.
  2. Classify and self-test the remaining confirmatory models; fill the placeholders in logs/prereg_deep_look_write.md
     and commit it (before any arm is fitted on seeds 1-4).
  3. 200M-token passes and fit_arms_deep on attn2L_s1..s4 and attn4L_s1..s4; summarize_deep.py; code_test.py on the
     held-out heads; an independent re-run of one model per depth; write logs/deep_look_write_results.md.

## Stopped 2026-10-15 10:57 (by request)

- Trained: attn2L_s0 2.2751, attn2L_s1 2.2665, attn2L_s2 2.3310, attn4L_s0 2.1273. attn2L_s3 stopped at step 9350
  (resume state saved 10:56). Still to train: attn2L_s3 (rest), attn2L_s4, attn4L_s1..s4.
- CPU: the first queue finished (B, C, 20M-token code-check passes and arm fits on attn_s1..s4). Addendum D (echo
  trajectory) had just started its first checkpoint pass; it restarts from scratch for that checkpoint.
- To continue: `sh scripts/resume_deep_study.sh` (training queue + watchdog + CPU queue; finished work is skipped).

## 2026-10-16 essay read verbatim (step 0 of the reprioritised plan)

Cammarata, N., 2026, "Neural Networks Are Already Clean", Goodfire (https://already-clean.goodfire.com/), read in full
from the downloaded page (2,247 lines of text). Quotes used for the reprioritisation, by section:
- "Why Networks Generalize": "circuits race to capture the remaining error, and the ones that reuse what's already
  there, and can be reused in turn, outcompete the rest. Generalization is what's left from the victor."
- "Where Adversarial Examples Go": "established routes have a head start in capturing residual error. They already
  provide ways to change the answer, so optimization can build on them."
- "More on Circuit Ecology": "A region of the network that already has useful parts and readers can learn the next
  related thing faster than anything starting fresh, so it takes the error signal before a competitor can grow."
- "Superposition of the Gaps": recursive generalization "is easy to check if you have training checkpoints."
- "Attention": "We've made less progress on attention than on neurons"; "Where it looks is relatively easy. It
  basically comes for free" (meaning: readable in attribution graphs, not explained); "My low-confidence prediction is
  that attention heads will turn out to be broad and recursively generalized, like neurons in small models, since
  there are far fewer of them."
- "Relation to Attention Superposition": "We think they meant that useful things are shared by groups of attention
  heads, and we agree"; "within a layer, two or three heads do most of a route's work".
- "Toy Models": the toy setting "prices the cost of sharing a neuron, interference, but not the benefit, reuse."
- The Introduction and "Exploring Image Models": with less room, units get "coarser-grained", not packed.

Reading for this study: our usefulness (how much of the residual error, relative to the bigram stage, a head's
existing write can capture) is a quantitative version of "established routes have a head start in capturing residual
error", applied to where a head looks. The essay does not test attention; its attention claims are low-confidence.
The growth test (write before look; attention acquired where the vote is useful) and the capacity comparison (fewer
heads -> broader, more shared work) follow from these passages. Correction to my earlier summary: the essay frames
shared work among heads as "groups of attention heads", not as superposition.

## 2026-10-16 echo test started first (MPS); capacity on development models (exploratory)

- pass_bgu on MPS equals CPU (max relative difference 2e-6 over all quantities, attn_s0, 1M tokens).
- Generic earlier-token ablation (common_deep.forward ablate_ctx) reproduces harness.common's "earlier" ablation
  exactly on attn_s0 (same head means; KL equal to 5 decimals for all 4 heads).
- Capacity (analysis/look_write_deep/capacity.py; exploratory; one-layer models and development seeds only, before the
  deep pre-registration): single-head earlier-token ablations, pairwise probability-weighted correlation of their
  effects.
  | model | heads | share of pairs with rho < -0.1 | mean rho | same-layer mean rho |
  |---|---|---|---|---|
  | attn_s0..s4 | 4 | 0.33-0.83 | -0.08 to -0.10 | (all same layer) |
  | attn2L_s0 | 8 | 0.00 | +0.046 | -0.013 |
  | attn4L_s0 | 16 | 0.00 | +0.049 | -0.005 |
  The one-layer heads correct each other (anti-correlated effects); in the deeper models even same-layer heads (also 4
  per layer) are uncorrelated. Direction as the essay predicts (fewer heads -> broader, shared work). One development
  model per depth so far; seeds 1-4 follow after the deep pre-registration.

## 2026-10-17 calibration, FINAL pre-registration, confirmatory runs started

- All 10 deep models trained: attn2L_s0..s4 dev 2.275 / 2.266 / 2.331 / 2.272 / 2.268; attn4L_s0..s4 2.127 / 2.146 / 2.132 /
  2.142 / 2.133. All pass the selftests. Confirmatory heads (seeds 1-4): F1 11, F2 16, F3 21.
- pass_bgu batched over heads (equal to the per-head version to 2e-7); MPS throughput: 4 blocks ~287 s per 10M tokens.
- Calibration (attn_s1..s4, 200M tokens): R1 -> LTOTAL (median own gain 0.104 vs TOTAL 0.049); R2 -> theta 0.20 (median
  own/oracle ratio 0.372); R3 -> c1 0.243, c2 0.271 (analysis/out/look_write_deep/rules.json).
- logs/prereg_deep_look_write.md committed FINAL (9c149a1; descriptive correction 8ab4a9f) before any confirmatory fit.
- Running unattended: scripts/confirmatory_passes.sh (8 models, ~8.5 h MPS) and scripts/confirmatory_fits.sh (fits as
  passes finish, then summarize_deep, code_test, capacity on seeds 1-4). Remaining by hand: independent re-run of one
  model per depth, write-up (logs/deep_look_write_results.md).
