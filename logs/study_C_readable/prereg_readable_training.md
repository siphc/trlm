# Pre-registration (FINAL): can a language model be trained to be readable?

FINAL, 2026-10-18; the DRAFT was committed 2026-10-17 (fce083b) before any output of this study existed. This version
adds the values fixed by the decision rules R1-R7 (from development outputs only), what has been seen, and the
deviations. It is committed before any confirmatory model (seeds 1-4) is trained. Decision: logs/decisions.md item 9.
Configuration: configs/side_readable.yaml. Running log: logs/readable_training.md.

## What has been seen before writing this

- Program v1.3 (commit 573c4ac) is 0.5497 nats from attn_s0 on dev_fast, and 0.540-0.561 from attn_s1..s4 with
  unchanged scalars (logs/mech_v3_report.md). Another seed of the same network is 0.38-0.42 from attn_s0 (TRAIN).
- In attn_s0, exact per-head tables help only with exact attention and vice versa; swapping one head's corpus table
  into the exact model costs 0.17-0.73 nats (logs/token_geometry_results.md).
- Nothing about models trained with a readability loss.

## Values fixed by the rules (analysis/out/readable_training/rules.json)

| rule | value | from |
|---|---|---|
| R1 | k_out 256; k_h 16 (LOCAL-0), 128 (RECALL), 16 (LOCAL-2) | coverage 0.973 / 0.973 / 0.993 / 0.974 on 256 pool windows |
| R2 | c = 0.836 | attn_s0 on the 512 held-out pool windows: L_att 0.559, L_vote 0.668 |
| R3 | **lambda_O = 0.3, lambda_I = 0.3** | grid cost O 0.3 +0.32%, 1 +3.1%; I 1 +0.23%, 3 +4.2%; seed-0 one epoch: O 0.3 +0.81%; I 1 +3.74% (> 2%: stepped down), I 0.3 +1.10% |
| R4 | sigma: harness KL 0.0093, OOD KL mean 0.0086, lock 0.035, attention-pattern KL 0.20 (out of distribution 0.20), vote correlation uniform 0.021 / weighted 0.043, ablation shifts 0.84 / 0.11 / 1.24 / 0.29 (heads 0-3) | aligned attn_s1..s4 |
| R5 | lambda ramps linearly over the first 5% of steps, both arms | see deviations |
| R6 | 8 auxiliary windows per step | with 16: arm O 1.9x, arm I 2.9x the baseline step |
| R7 | pool stays at 32,768 windows | seed-0 memorisation gaps: O 1.0%, I -0.7% of the held-out value |

Confirmatory runs: arms O and I at lambda 0.3, seeds 1-4, one epoch, 8 auxiliary windows, 5% ramp; then the suite
(analysis/readable_training/suite.py with --lock --harness --refit) on each, against the twins attn_s1..s4.

## What has been seen (development only)

- **Grid** (seed 0, 100M tokens; logs/readable_training.md): arm O lowers KL(model || program) from 0.61 to 0.25 as
  lambda grows, at a cost of up to +8.9%; at lambda 3-10 attention moves away from the rules (pattern KL 1.7-1.9 vs 0.87).
  Arm I lowers attention-pattern KL from 0.87 to 0.06 and raises the vote correlation from 0.27 to 0.71, while KL to the
  program stays at 0.51-0.58; at lambda 0.1-0.3 it lowers dev cross-entropy by 2-3% at 100M tokens. The positive
  control (256-window pool) has a memorisation gap of 0.158; all other runs <= 0.005.
- **Seed-0 one-epoch runs against attn_s0** (no verdict weight; seed 0 shares the target's model):
  arm O (0.3): harness KL 0.550 -> 0.396, out of distribution 0.810 -> 0.604, attention-pattern KL 0.475 -> 0.441, vote
  correlation 0.29 -> 0.31, lock 0.50 -> 0.49. Arm I (0.3): harness KL 0.550 -> 0.518, out of distribution 0.810 ->
  0.854, attention-pattern KL 0.475 -> 0.201 (out of distribution 0.491 -> 0.216), vote correlation 0.29 -> 0.53, lock
  0.50 -> 0.55. Arm I at lambda 1: RECALL's ablation shift fell to 0.46 of attn_s0's (a decorative-head flag).
- **Baselines** attn_s0..s4: harness KL 0.540-0.561; claims passed on aligned checkpoints 7, 6, 4, 6, 7 of 7;
  refitted-program KL 0.543-0.581 (refitting does not help).

## Deviations and clarifications

- R1's two conditions conflicted (95% coverage needs 13.5 GB, above the 10 GB cap); resolved in favour of coverage before
  the pool was built.
- R5 fires on batch difficulty: the lambda = 0 control "spikes" at the same steps as every run (all seed-0 runs see the
  same batches). Applied as written (a 5% ramp, harmless); recorded as a defect of the rule.
- R3's second part was applied iteratively (step down, rerun, re-check), as its text implies.
- A measurement bug was fixed before any confirmatory output: refitted program copies were written where the harness
  sandbox cannot read them; they now live in programs/readable/ and the refit scores (b2) were redone for every model.
- Closures use the twin's values from the same tool, windows and code (analysis/out/readable_training/attn_s{s}/
  suite.json); trained closure for arm I uses the attention-pattern KL (d3) and the row-weighted vote correlation (d4);
  untrained for arm O uses d3 and the uniform vote correlation.

## Models

1-block attention-only, configs/frozen.yaml v3 otherwise; one epoch (527,482,880 tokens), peak LR 2e-3. Seed s uses
the initialisation and data order of attn_s{s} (its untreated twin). Seed 0: development only (the program was fitted
to attn_s0). Confirmatory: seeds 1-4, arms O and I at their chosen strengths (8 models). Development: seed 0 at 100M
tokens: lambda = 0 and both arms at lambda in {0.1, 0.3, 1, 3, 10}; the positive control; arm I attention-only and
vote-only; then both arms at one epoch.

## Training signal

Each step adds a loss on 16 [R6] auxiliary windows from a fixed pool of 32,768 TRAIN windows (default_rng(9001);
windows of every analysis set listed in configs/side_readable.yaml excluded; 512 further windows held out). Pool
targets are program v1.3's outputs, precomputed and checked against a fresh harness setup.
- **Arm O:** CE + lambda_O * KL(program || model) over the partition {top-k_out [R1] tokens of the program, rest}.
- **Arm I:** CE + lambda_I * (L_att + c [R2] * L_vote).
  - L_att: mean over model heads 0, 1, 2 of KL(rule || model) over {top-k_h [R1] keys, rest}, rules LOCAL-0, RECALL,
    LOCAL-2 (program/head0.py, head1.py, head2.py), query rows i >= 1.
  - L_vote: for heads 0, 1, 2, the weighted mean over key tokens s of 1 - rho_h(s), where rho_h(s) is the
    unigram-weighted correlation over v between the head's OV table row OV_h(s, .) (direct path through the
    unembedding, final LayerNorm gain, scale-free) and its target row. Targets: head 0 = LOCAL-0's earlier-token vote
    averaged over its own distance profile; head 2 = the same for LOCAL-2; head 1 = RECALL's copy/topic/next vote for
    the token's group. Row weights: half how often the rule reads s (pool), half the unigram frequency of s.
  - Head 3 and the direct path are unconstrained.

## Tools (per model; "trained" = the arm's loss optimises this quantity or its other direction)

| id | tool | arm O | arm I |
|---|---|---|---|
| b1 | harness KL(model ‖ program v1.3), dev_fast | trained (other direction) | untrained |
| b2 | harness KL after refitting the program's scalars to the model (scripts/fit_scalars.py, one round) | partly | partly |
| c | KL(model ‖ program) on out-of-distribution windows: the twin's probe samples (data/eval/probe/attn_s{s}.npy), 200 program-generated windows, sentence-shuffled and word-shuffled dev_full windows 200-399 | untrained | untrained |
| c' | memorisation gap: trained loss on 64 held-out pool windows minus on 64 pool training windows | | |
| d1 | claim tests (program/tests) on the head-aligned checkpoint | untrained | untrained |
| d2 | the lock: mean over heads 0-2 of KL(swap head h's attention and table for the program's, everything else exact) - KL(all exact), analysis/token_geometry/tables_vs_attention.py | untrained | untrained |
| d3 | attention pattern KL(model head ‖ rule), heads 0-2, train_windows(16, 9125) | untrained | trained (other direction) |
| d4 | OV row correlation with the vote targets, uniformly weighted | untrained | trained with other weights |
| d5 | per-head mean-ablation shift KL(model ‖ model with head h at its mean), heads 0-3 | untrained | untrained |
| e | seed distances among the new models and to attn_s0..s4 | untrained | untrained |

Heads are aligned before measuring: model heads are permuted (an exact symmetry, checked to |delta logits| < 1e-4) so
that the heads playing LOCAL-0, RECALL and LOCAL-2 (best total pattern KL with the v1.3 rules) are heads 0, 1, 2.

## Definitions

- **Closure** of a tool for a new model against its twin attn_s{s} (same tool, same windows): (base - new) / base for
  KL-type tools; (rho_new - rho_base) / (1 - rho_base) for correlations. An improvement smaller than 2 sigma_T [R4]
  counts as closure 0.
- **Trained closure:** O: closure on b1. I: mean of the closures on d3 and on the weighted vote correlation.
- **Untrained closure (primary):** O: mean of the closures on d3 and d4. I: closure on b1.
- **Transfer** tau = untrained closure / trained closure.
- **OOD transfer** tau_OOD: O: mean closure on the four sets of c / closure on b1. I: closure of d3 computed on the
  four OOD sets / closure of d3 on train_windows(16, 9125).

## Decision rules (applied to development outputs only; values go into FINAL)

- R1: k_out = the smallest of {64, 128, 256}, and k_h the smallest of {8, 16, 32, 64, 128} per rule, whose stored
  entries carry on average >= 0.95 of the program's probability (attention mass), on 256 pool windows; total storage at
  most 10 GB.
- R2: c = L_att / L_vote measured on attn_s0 (aligned) on the 512 held-out pool windows, so the two terms are equal
  there.
- R3: per arm, lambda* = the largest grid value whose 100M-token run has dev_full CE <= 1.01 x the lambda = 0 run's
  (0.1 if none). If the seed-0 one-epoch run at lambda* costs more than 2% dev_full CE against attn_s0, step down one
  grid value (and repeat the seed-0 run).
- R4: sigma_T = the standard deviation of tool T over the aligned attn_s1..s4.
- R5: if any run shows a loss spike (a rise of more than 0.1 nats within 200 steps after warmup), all runs of that arm
  ramp lambda linearly over the first 5% of steps.
- R6: 16 auxiliary windows per step, 8 if a step of either arm takes more than 1.6 x the baseline step on MPS.
- R7: if the seed-0 one-epoch run's memorisation gap (c') exceeds 10% of its held-out value, the pool grows to 65,536
  windows (appended) before confirmatory runs.

## Hypotheses

- **H-genuine:** the pressure moves the model toward the program's way of computing. Agreement transfers to tools,
  windows and inputs that were not trained on; the constrained heads keep doing the work; the lock loosens; seeds
  become more alike.
- **H-Goodhart:** the trained quantity improves by other means. Untrained tools stay near baseline, agreement fades
  out of distribution or stays on the pool, constrained heads become decorative (work moves to head 3 or the direct
  path).
- **H-capacity:** the pressure barely moves the trained tool, or costs much prediction quality.

## Tests per confirmatory seed (1-4), per arm

- **M (manipulation check):** trained closure >= 0.3.
- **P (primary):** tau >= 0.5 counts for H-genuine, tau <= 0.2 for H-Goodhart.
- **S (out of distribution):** tau_OOD >= 0.5 genuine, <= 0.2 Goodhart.
- **Goodhart flags:** (i) memorisation gap > 0.02 nats; (ii) any of heads 0-2 has a mean-ablation shift below 50% of
  its twin's role-matched head, or head 3's shift exceeds 1.5 x its twin's; (iii) b2 is worse than b1 by more than
  0.05 nats (refitting the program to the model's own heads hurts: its heads do not look or vote as the program says).

## Verdict per arm

- INCONCLUSIVE if M fails for 2 or more seeds.
- GENUINE if P and S both reach 0.5 in at least 3 of 4 seeds and no flag occurs in more than 1 seed.
- GOODHARTED if P <= 0.2 in at least 3 of 4 seeds, or S <= 0.2 in at least 3 of 4, or one flag occurs in at least 3
  of 4 seeds.
- PARTIAL otherwise.

## Reported separately (no verdict weight)

- Cost: paired difference in dev_full CE (mean +- SE over seeds 1-4).
- Fidelity achieved: b1 and b2 per model.
- The lock (d2): LOOSENED if it drops by at least 30% against the twin.
- Seed convergence: relative drop of the mean pairwise KL among the arm's seeds 1-4 (vs among attn_s1..s4), divided
  by the relative drop of the mean KL to the program; >= 0.5 converge, <= 0.2 not.
- attn_s0 specificity: mean KL(new ‖ attn_s0) - mean_k KL(new ‖ attn_sk), k = 1..4 (expected <= 0.02).
- **Positive control:** arm O, lambda = 10, a 256-window pool, seed 0, 100M tokens. It must show flag (i) and
  tau_OOD <= 0.2; if it does not, the suite is declared insensitive in the results.

## Recorded expectations (no verdict weight)

Arm O closes much of b1 at little cost (the program's dev cross-entropy already equals the model's) with partial
internal transfer. Arm I matches attention easily, transfers little to the output, and risks decorative heads at high
lambda.

## Exploratory

The development grid's dose-response curves; arm I attention-only and vote-only; refits beyond one round; the new
models' own samples as a fifth OOD set; no use of the test split.

## Amendment (2026-10-18, before any confirmatory output was read)

- **Test M lowered from 0.3 to 0.2** (human decision, logs/decisions.md item 11). Reason: the seed-0 one-epoch runs
  (development) have trained closures 0.28 (arm O) and 0.274 (arm I) although the trained tools moved far beyond noise
  (arm O's harness KL by about 15 sigma); these values were in hand when the FINAL was written but were not checked
  against M. Based on development outputs only; the confirmatory runs had started, no confirmatory output had been read.
  The verdict with M at 0.3 is reported next to the amended one.
- Clarifications of ambiguous wording (noise floor, undefined ratios, flag counting, the positive control's twin and
  others) are listed in logs/readable_training.md, phase 4; all are implemented in analysis/readable_training/verdicts.py.
