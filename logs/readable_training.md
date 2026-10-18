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

## 2026-10-18 phase 3: development grid (seed 0, 100M tokens; analysis/out/readable_training/grid_table.json)

| run | dev_full CE | cost | KL(model‖prog) | KL(prog‖model) | attn KL(model‖rule) | attn KL(rule‖model) | vote ρ (uniform, mean h0-2) | memo gap out | memo gap attn |
|---|---|---|---|---|---|---|---|---|---|
| I_l0.1_s0_100M | 2.5786 | -3.13% | 0.531 | 0.649 | 0.357 | 0.240 | 0.456 | 0.000 | -0.000 |
| I_l0.3_s0_100M | 2.6066 | -2.08% | 0.524 | 0.623 | 0.255 | 0.132 | 0.532 | 0.003 | 0.000 |
| I_l10_s0_100M | 2.9257 | +9.91% | 0.581 | 0.581 | 0.059 | 0.022 | 0.712 | 0.004 | 0.000 |
| I_l1_s0_100M | 2.6680 | +0.23% | 0.510 | 0.585 | 0.165 | 0.067 | 0.607 | 0.002 | 0.000 |
| I_l3_s0_100M | 2.7747 | +4.24% | 0.519 | 0.568 | 0.107 | 0.037 | 0.670 | 0.002 | 0.000 |
| O_l0.1_s0_100M | 2.6647 | +0.11% | 0.544 | 0.591 | 0.924 | 0.881 | 0.274 | 0.002 | 0.008 |
| O_l0.3_s0_100M | 2.6704 | +0.32% | 0.465 | 0.477 | 0.775 | 0.741 | 0.269 | 0.001 | 0.004 |
| O_l10_pool256_s0_100M | 3.0314 | +13.88% | 0.383 | 0.351 | 2.136 | 1.344 | 0.227 | 0.158 | 0.046 |
| O_l10_s0_100M | 2.8977 | +8.86% | 0.248 | 0.229 | 1.907 | 4.582 | 0.264 | 0.003 | 0.003 |
| O_l1_s0_100M | 2.7445 | +3.10% | 0.349 | 0.334 | 0.765 | 0.718 | 0.266 | 0.002 | -0.000 |
| O_l3_s0_100M | 2.8378 | +6.61% | 0.279 | 0.258 | 1.665 | 2.659 | 0.173 | 0.002 | 0.026 |
| none_s0_100M | 2.6618 | +0.00% | 0.608 | 0.696 | 0.869 | 0.832 | 0.274 | 0.000 | 0.006 |

(Online metrics at the last evaluation, on 64 held-out pool windows; attention KL is the mean over heads 0-2; the
"memo gap" is the trained loss on held-out minus pool-training windows.)

- **Rule R3:** lambda_O* = 0.3 (cost +0.32%; 1 costs +3.1%); lambda_I* = 1 (cost +0.23%; 3 costs +4.2%).
- **Rule R5:** fired in every arm, but the lambda = 0 control "spikes" at the same steps (1450, 5200-5600): all seed-0
  runs see the same batches, and some batches are harder. The rule as written measures batch difficulty, not
  instability (a defect: it should have compared against the control). Applied as written: lambda ramps linearly over
  the first 5% of steps in both arms (harmless). Recorded as a deviation-free application of a poorly calibrated rule.
- **Positive control** (arm O, lambda 10, 256-window pool): memorisation gap 0.158 (flag threshold 0.02; all other runs
  <= 0.005). The memorisation check is sensitive.
- **Development observations (one seed, no verdict weight):** (1) arm I at small lambda lowers dev cross-entropy
  (-3.1% at 0.1, -2.1% at 0.3): the program's attention rules and corpus votes act as a useful head start at 100M
  tokens; (2) strong output pressure moves attention away from the rules (attention KL 1.7-1.9 at lambda_O 3-10 vs 0.87
  in the control) while output agreement improves (KL 0.25-0.28): the Goodhart signature on the development seed;
  (3) moderate output pressure (0.3-1) moves attention slightly toward the rules (0.87 -> 0.77) and leaves the vote
  correlation unchanged; (4) strong internal pressure barely moves output agreement (KL 0.51-0.58 at lambda_I 1-10).
- **Arm-I extras** (seed 0, 100M tokens, lambda 0.3; exploratory): attention-only: dev CE -2.46%, attention-pattern KL
  0.185, vote correlation 0.322, KL to the program 0.537; votes-only: -1.48%, 0.848, 0.541, 0.555; both (I_l0.3):
  -2.08%, 0.255, 0.532, 0.524; control: 0.869, 0.274, 0.608. The two internal parts are separable: training one does
  not move the other, and each alone improves prediction at 100M tokens.

## 2026-10-18 phase 3b: seed-0 one-epoch runs, rules R3 (second part) and R7, FINAL pre-registration

- R3, second part: arm O at 0.3 costs +0.81% against attn_s0 (2.5297 vs 2.5095): kept. Arm I at 1 costs +3.74%
  (2.6032): stepped down to 0.3, rerun: +1.10%: kept. Final lambda_O = lambda_I = 0.3. (Arm I's head start at 100M tokens
  turns into a small cost by the end of the epoch.)
- R7: memorisation gaps 1.0% (O) and -0.7% (I) of the held-out values: no pool growth.
- Measurement fix: the harness sandbox cannot read analysis/, so the refitted programs (tool b2) failed to import for
  every model; moved to programs/readable/ (gitignored) and re-scored. Refitting never helps: attn_s1..s4 0.552, 0.581,
  0.543, 0.549 (unchanged program 0.547, 0.561, 0.540, 0.545); seed-0 arm O 0.3965 (0.396).
- Development preview against attn_s0 (seed 0; no verdict weight):

  | | attn_s0 | O (0.3) | I (1) | I (0.3) |
  |---|---|---|---|---|
  | harness KL to program | 0.550 | 0.396 | 0.503 | 0.518 |
  | OOD KL to program (mean of 4 sets) | 0.810 | 0.604 | 0.834 | 0.854 |
  | attention-pattern KL to the rules | 0.475 | 0.441 | 0.146 | 0.201 |
  | same, out of distribution | 0.491 | 0.444 | 0.160 | 0.216 |
  | vote correlation (weighted) | 0.332 | 0.367 | 0.785 | 0.698 |
  | lock | 0.501 | 0.488 | 0.497 | 0.551 |
  | ablation shift RECALL head | 0.452 | 0.327 | 0.207 | 0.353 |
  | claims passed (of 7) | 7 | 6 | 7 | 7 |

  Each arm robustly changes the level it is trained on (including out of distribution) and barely moves the other: the
  program's behaviour and its mechanism description are decoupled in this model. The lock does not loosen.
- FINAL pre-registration written with all rule values; waiting for gate G2 (dose-response and lambda) before the
  confirmatory runs (arms O and I at 0.3, seeds 1-4: about 8 h of MPS).

## Phase 4: confirmatory runs (2026-10-18)

- Gate G2 approved by the human on 2026-10-18 (after the plan for the next studies, decision 10; these runs are part A
  of study C in logs/plan_attention_law_and_readability.md). Started `scripts/readable/confirm_queue.sh` (arms O and I
  at lambda 0.3, seeds 1-4, one epoch each, MPS) with the battery watchdog; the suite (`--lock --harness --refit`) runs
  on each model in the background as it finishes (logs/runs/readable_suite_*.log).
- **Clarifications of the FINAL pre-registration, written before any confirmatory output** (implemented in
  `analysis/readable_training/verdicts.py`; no rule value changes):
  1. Noise floor: a tool's change against the twin (raw scale, sign such that positive = toward the program) counts
     as closure 0 when its size is below 2 sigma_T; a worsening of at least 2 sigma_T keeps its negative closure (a
     model that moves away from the program's description is evidence against H-genuine). The literal alternative
     (every change below +2 sigma_T, including all worsenings, counts as 0) is reported next to it.
  2. tau and tau_OOD are undefined, and count toward neither threshold, when their denominator closure is <= 0.
  3. tau_OOD numerators: arm O, the mean over the four out-of-distribution sets of each set's closure (each with its own
     sigma); arm I, the closure of the pattern KL averaged over the four sets (sigma of that average), as worded.
  4. Flags are counted per flag: "no flag occurs in more than 1 seed" means each of (i), (ii), (iii) occurs in at most
     one seed; the number of seeds with any flag is reported as well.
  5. Flag (i) uses the last logged evaluation of the trained loss (arm O output, arm I attention), as R7 did.
  6. Seed convergence uses tool e's windows (train_windows(64, 9100)) for both distances: the mean pairwise KL over
     ordered pairs among the arm's seeds 1-4 vs among attn_s1..s4, and the mean KL to the program on the same windows.
  7. attn_s0 specificity as worded (the mean over k = 1..4 includes the twin); also reported without the twin.
  8. Positive control: its twin is the lambda = 0 run with the same tokens and batches (none_s0_100M); attn_s0 as an
     alternative; b1 from the harness for both.
  9. A noted risk, no change: sigma of the pattern KL (0.20) is large; the twins' values are 0.585-1.038, so arm I's
     pattern closure counts only if it improves by more than 0.40 nats (for attn_s3's twin, from 0.585 to below 0.18).
     If it does not, arm I's trained closure rests on the vote correlation alone and may fall below the 0.3 of test M.
- **Found while checking verdicts.py on development models (before any confirmatory output):** the seed-0 one-epoch
  runs fall just below test M (trained closure >= 0.3) under the FINAL rules: arm O 0.28 (harness KL 0.550 -> 0.396, about
  15 sigma), arm I 0.274 (the pattern-KL improvement 0.475 -> 0.201 is below the 2 sigma floor of 0.40 and counts as 0;
  vote correlation closure 0.55). These values were available when the FINAL was written and were not checked against
  M (an oversight). If seeds 1-4 behave alike, arm O is INCONCLUSIVE by rule. Rules unchanged; the human was asked
  whether to amend M before any confirmatory output is read. Other seed-0 values (no verdict weight): arm O tau 0
  (no internal closure beyond 2 sigma), tau_OOD 0.94; arm I tau 0.21, tau_OOD undefined; no flags.
