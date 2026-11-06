# Mechanism program, round 3 (v1.2 -> v1.3), 2026-10-17

Where the program differs from the model, how far apart seeds of the same network are, and RECALL's fixed-size
vote.

## Assumptions

- Starting point: program v1.2 (commit 1ef9b7a), dev_fast KL 0.5706, top-1 0.633, claims 6/6.
- Rules unchanged: "explain, don't fit" (logs/decisions.md items 5-7). Token-level knowledge only from corpus
  statistics counted in `setup(corpus)` plus named scalars. Decision 8: a per-token corpus code is an analysis test
  only, not a program component.
- Adoption bar unchanged (decision 3): paired held-out gain >= 0.005 and >= 3 SE (>= 0.01 for a new component).
- All numbers below are on TRAIN windows (analysis/fitlib.train_windows; seed 2000 = 64 eval windows unless stated),
  never dev or test. Scripts: analysis/program_gap/ and analysis/scratch/program_gap/; outputs analysis/out/program_gap/.

## 1. Where the remaining KL is (diagnosis of v1.2)

- **The program is too flat.** Mean entropy 3.15 nats vs the model's 2.54; flatter at 82% of positions. It puts 63%
  of its mass on the model's top-10 tokens, the model 78%; 0.45 of the 0.57 nats come from the model's top-10
  tokens. The best global temperature is 1.0 (the fitted strengths already set it); a per-position oracle
  temperature reaches 0.49. Over the whole vocabulary the program's scores spread twice as wide as the model's
  logits (std 9.4 vs 5.2); among frequent tokens (frequency-weighted) less wide (3.0 vs 4.7): too harsh in the tail,
  too timid at the top (analysis/program_gap/diagnose.py).
- **KL is spread evenly over positions** (about 0.55 everywhere; the first 4 positions of a window add only about
  0.0075 nats in total) and over token shapes (word starts 65% of the KL at 67% of positions; continuation pieces
  21% at 16%).
- **The program's parts are co-adapted, not part-faithful** (analysis/program_gap/budget.py; exact additive split of the
  model's logits with the final LN scale fixed). Swapping one program group into the otherwise exact model (each
  part's position-constant vector moved to a shared bias): current-token group (FOLLOW + self votes) 0.40, local
  group (LOCAL-0/2 earlier + PHRASE) 0.62, RECALL 0.29, bias vector only 0.21; all together 0.56. Using the model's
  bias in the program makes it worse (0.90). Earlier oracles say the same for heads 0 and 2: exact vote tables with
  the program's attention are worse (+0.038), exact attention with corpus votes gains 0.006, both exact 0.060.
- **The single largest identifiable gap is the model's current-token table** (analysis/program_gap/follow_gap.py). With
  every head's earlier-token part exact, the program's current-token directions (refitted) reach 0.332; a
  context-free oracle table (the model's own current-token logits averaged per token; insight only) reaches 0.072.
  So the model's current-token contribution is almost entirely a per-token table, and the corpus bigram is a poor
  stand-in for it.
- **That table is about two-thirds corpus statistics** (analysis/program_gap/table_probe.py; 282 tokens with >= 40
  positions, model-mass weighted): R^2 0.61 from log P1 alone, 0.63 with the unigram, 0.66 with copy, slot and
  successor similarity, skip-gram, reverse bigram, continuation count and word-piece flag. Projecting the bigram
  through the model's own 128-dimensional embedding span lowers it (0.32).
- **The model is not closer to the corpus equilibrium than the program** (analysis/program_gap/moments.py, 256 windows):
  its mean prediction after each token is further from the corpus bigram (0.102) than the program's (0.078). So
  calibrating the program to corpus moments would move it away from the model.
- **The model's token code is not a closed-form corpus construction** (analysis/scratch/program_gap/span_theory.py):
  the leading directions of the embedding match corpus structure (top-32 canonical correlations 0.8-0.9,
  frequency-weighted), but the bulk does not (mean squared canonical correlation over 128 dimensions 0.14-0.41 for
  spectral spans of the bigram, PMI and the symmetrised co-occurrence matrix that small-initialisation training
  theory points to; random 0.07-0.23).

**Reading.** Under the current rules the program is close to its ceiling for the local heads (logs/
mech_v2_investigations.md) and for the current-token prediction (this file). What is missing is the model's learned
token geometry: how it smooths each token's predictions over tokens its embedding treats as alike. No corpus
construction tried reproduces it. Further large gains need a decision by the human (section 4).

## 1b. How far apart are seeds? (analysis/program_gap/seed_floor.py, ensemble_target.py; 64 TRAIN windows)

| from attn_s0 | KL |
|---|---|
| attn_s1, s2, s3, s4 (same network and data, other seed) | 0.420, 0.424, 0.384, 0.387 |
| average of seeds 1-4 (probabilities) | 0.248 |
| mlp_s0 | 0.582 |
| program v1.2 | 0.564 |

- The program's data cross-entropy (2.501) equals attn_s0's (2.506); seeds 2.493-2.523.
- The program is equally far from every seed (KL(seed k || program) 0.557-0.574) although it was fitted to attn_s0
  only: it describes what the seeds share, nothing seed-specific. Refitting all its scalars to attn_s1 gives 0.569 on
  attn_s1 (seed-0 scalars: 0.561).
- Against the five-seed average ENS, single seeds are 0.18-0.20 away and the program 0.43. So of the program's 0.56
  nats to attn_s0, roughly 0.2-0.25 is seed-specific (no seed-agnostic explanation can remove it) and roughly 0.3 is
  shared behaviour the program misses.
- The model's current-token table (section 1): about 15% of its variance is seed-specific, about 20% shared by all
  seeds but not explained by the corpus bigram (analysis/program_gap/table_seeds.py).

**Consequence for the target.** Fidelity to attn_s0 is bounded below by its seed-specific part. "What all seeds agree
on" is the natural target for an explanation; KL(ENS || program) is reported next to the harness metric from now on
(analysis only; the harness is unchanged).

## 1c. Consensus failures and RECALL saturation (analysis/program_gap/consensus.py, ln_rule.py, vote_fit_sat.py)

- Positions where the five seeds agree but the program does not: 58% of the program's miss (KL(ENS || program)) is at
  positions whose consensus top token occurred earlier in the window (61% of positions): copying.
- Worst cases were RECALL over-copying with near-certainty: at the first position of a window ('air' -> 'air' 0.99;
  the seeds give it 0.06) and early in a window ('Jane and' -> ' Jane' 1.00), where RECALL's attention falls on one
  or two tokens and casts their full vote (copy + topic: +14 to +18 nats).
- **Mechanism: RECALL's summed vote saturates**, divided by sqrt(1 + c spread^2) (spread: unigram-weighted standard
  deviation over the vocabulary). All strengths refitted, 16 fit / 32 eval windows: -0.0162 +- 0.0017 nats (c 3.6;
  RECALL's strengths grow about fivefold). Same rule for LOCAL-0 -0.0020, LOCAL-2 -0.0023, PHRASE 0.0000; shrinking
  the whole score by RECALL's size instead -0.0048. With FOLLOW's piecewise response on top: -0.0196 (+0.0034, below
  the bar).
- **In the model** (analysis/program_gap/saturation_check.py, 16 windows): head 1's logit contribution grows much less than
  RECALL's raw vote. From the 9th to the 10th decile of the raw vote size, the raw size rises 42% and head 1's
  contribution 6%; at window position 0 the raw vote is 2.5 times its average size, head 1's contribution 0.7 times.
  By head 1's own attention concentration its contribution rises, then flattens and falls in the top decile.

## 2. Candidates tried this session (analysis/vote_fit.py, all strengths refitted; 32 fit / 64 eval windows)

| candidate | idea | paired gain |
|---|---|---|
| follow_pwl | FOLLOW responds piecewise-linearly to log P1 (steeper for frequent continuations) | -0.0045 +- 0.0003 (near miss) |
| shape_gates | FOLLOW/PHRASE strengths by current-token shape | -0.0018 +- 0.0002 |
| kn_bias | Kneser-Ney continuation preference added | -0.0005 +- 0.0002 |
| kn_follow | FOLLOW as an interpolated Kneser-Ney bigram | +0.0527 +- 0.0017 (worse) |
| sim_smooth | similarity-based smoothing of FOLLOW (Dagan, Lee & Pereira) | -0.0005 +- 0.0002 |
| softmax bottleneck | M-projection of the output onto a 128-dim span | oracle span (model W_E) 0.519 -> 0.477 (8 windows); corpus spans worse (0.54-1.28) |
| **RECALL saturation** | RECALL's summed vote divided by sqrt(1 + c spread^2) | **-0.0162 +- 0.0017 (16 fit / 32 eval); adopted as v1.3** |

## 2b. Program v1.3 (commit 573c4ac)

- RECALL writes a vote of fixed size (head1.py SIZE 0.525; the 15 strengths matter only as ratios). The saturating
  form sqrt(1 + c spread^2) fitted to its limit (strengths x1000 at the same c), so the program states the limit.
- Held-out TRAIN (64 windows, seed 2000): 0.5643 -> 0.5474. Harness dev_fast: **0.5497** (v1.2 0.5706), top-1 0.635,
  loss recovered 1.024 (the program's dev cross-entropy 2.511 is below the model's 2.535), claims 7/7 (new claim:
  where RECALL's raw vote is largest, head 1's effect grows less than that vote, and the fixed-size vote is closer).
- Ledger (64 TRAIN windows): FOLLOW +1.36, PHRASE +0.144, LOCAL-2 +0.134, RECALL +0.121, LOCAL-0 +0.066 (was 0.152).
- Against the five-seed consensus: KL(ENS || program) 0.427 -> 0.411.
- Next consensus failures: in-context completion and copying ('my cri' -> 'cket' 0.90-0.96 after "cricket" earlier;
  'Together, M' -> 'um'; '. We saved the park.' -> ' We'), and over-confident phrases near a window's start
  ('upon a' -> ' time' 0.99 at position 1; seeds 0.24).

## 2c. After v1.3

| candidate | paired gain vs v1.3 | verdict |
|---|---|---|
| RECALL token-level successor attention (branch `cand-recall-succ`) | -0.00453 +- 0.00057 (fresh), -0.00463 +- 0.00085 | near miss (bar 0.005) |
| FOLLOW piecewise response on top of the fixed-size RECALL | -0.0034 (vote_fit_sat) | near miss |
| window-start rule | at most about 0.004 nats per token available | not built |
| smoothing of unseen continuations | unseen tokens carry 0.055 of 0.55 nats | not the problem |

## 2d. v1.3 unchanged (seed-0 constants) on other models (harness dev_fast, no claims; logs/runs/score_v13_*.out)

| model | KL | top-1 | loss recovered |
|---|---|---|---|
| attn_s0 (fitted) | 0.5497 | 0.635 | 1.024 |
| attn_s1 | 0.5471 | 0.634 | 1.023 |
| attn_s2 | 0.5613 | 0.631 | 1.061 |
| attn_s3 | 0.5396 | 0.638 | 1.001 |
| attn_s4 | 0.5451 | 0.637 | 1.019 |
| mlp_s0 | 0.6206 | 0.611 | 0.806 |

The per-seed refit (scripts/seed_transfer.py) was stopped: with four jobs in parallel each model needed over 6 hours,
and refitting changes little (attn_s1: 0.561 with seed-0 scalars vs 0.569 refitted, TRAIN). To rerun overnight.

## 3. Running / next

- Phase 8 seed transfer of v1.2 (scripts/seed_transfer.py): attn_s1..s4 and mlp_s0, logs/seed_transfer_*.log.
- Next: the mlp variant (how much of mlp_s0 the attention-only program captures; whether the MLP is where phrase
  knowledge lives), then readability and human-judging preparation.

## 4. Question for the human (not blocking)

The remaining gap is the model's token geometry. Options: (a) accept the plateau for attn_s0 and finish phases 8-9
(seed transfer, mlp, refactor, judging); (b) allow a per-token code in the program. A corpus-derived code would not
help (section 1); a code fitted to the model is the distillation approach rejected in decision 5. So (a) is the
recommendation; (b) is listed only for completeness.
