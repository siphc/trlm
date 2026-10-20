# Pre-registration (FINAL): how attention forms when what a head writes is fixed (study A, `fixed_votes`)

FINAL, 2026-10-20. The DRAFT was committed 2026-10-18 (a7a9c5f) before any fixed-vote model was trained and before the
usefulness tables (the law's model-free prediction) were computed. This version adds the values fixed by the decision
rules (development outputs only, seed 0), what has been seen, and the deviations. It is committed before any
confirmatory model (seeds 1-4) is trained; the confirmatory runs wait for the human's gate G2. Decisions:
logs/decisions.md items 10 and 12. Plan: logs/plan_attention_law_and_readability.md section 4. Code:
src/train/fixed_votes.py (model), src/train/test_sanity_fixed_votes.py (9 tests), analysis/fixed_votes/ (usefulness.py
M2, law.py M1 and M3-M6, rules.py, verdicts.py: the tests below, refuses to run before this commit; check_pipeline.py).
Running log: logs/fixed_votes.md.

## What has been seen before writing the DRAFT

- The look-follows-write law in normal one-layer attention-only models (logs/look_follows_write_results.md): in 7 of 8
  local heads of attn_s1..s4, own-vote usefulness recovers at least half of the output gain of the exact token-token
  attention term (P1); another head's or a shuffled vote gives at most a quarter of that (P2, 8/8, at most 0.10); the
  pure-pair correlation is 0.34-0.44 for the own vote and at most 0.08 for other heads' (P3, 8/8); about 0.24 attention
  score per logit of usefulness (P4). For far-looking heads, usefulness at distances 1-7 recovers only 6-38%
  (long range not tested). One head had a negative ceiling (the ratio is then undefined). Against the stricter
  ceiling `tt_self` (the token-token term at every key including the current token; addendum,
  logs/look_follows_write_addendum_results.md) own usefulness recovers 26-46% of the gain.
- The fixed tables (data/fixed_votes/tables.npz, sha 135ef9807fb12aba): row means 0 and unigram-weighted RMS 1 by
  construction; value ranges next -1.5..9.2, copy -0.6..9.6, topic -27..25 (rare tokens), skip2 -2.2..10.3,
  skip3 -3.0..13.0. Not clipped.

## Values fixed by the rules and the model-free predictions (analysis/out/fixed_votes/rules.json, usefulness/profile.json)

| rule | value | from |
|---|---|---|
| R1 | **peak LR 4e-3** | dev_full CE after 50M tokens: 1e-3 2.6120, 2e-3 2.5874, 4e-3 2.5775 (the largest grid value; the optimum may lie higher) |
| R2 | measured on the instance at setup (binding there); laptop GPU 0.21 s/step (not binding) | |
| R3 | not applied (applies only if R2 fires on the instance) | |
| R4 | all four heads active in both development models (vote-removal shifts 0.060-0.481) | |
| R5 | defaults kept: band 8-63 (first 100M tokens), shrinkage pseudo-count 20, 32 fit and 32 evaluation windows | |
| M2 | **H3a predictions: skip2 peaks at d = 1, skip3 at d = 2** (mean usefulness d = 1..7: skip2 0.839, 0.261, 0.152, ...; skip3 0.277, 0.700, 0.301, ...) | computed from corpus counts after the DRAFT, before any model (commits 3ad0532, a22b1f6) |

**Confirmatory runs:** base and shift, seeds 1-4, one epoch each at LR 4e-3; law.py on all eight. All on the cloud
instance (NVIDIA L4) if it is available, otherwise all on the laptop GPU.

## What has been seen in development (seed 0, laptop GPU; no verdict weight)

| | base | shift |
|---|---|---|
| dev_full CE (attn_s0 2.5095) | 2.5428 (+1.3%) | 2.6213 (+4.5%) |
| vote strengths next, copy, topic, head 3 | 0.70, 0.68, 1.51, 1.78 | 0.83, 0.66, 1.33, 1.67 |
| own / tt (P1' needs >= 0.5): next, copy, topic, head 3 | 0.24, 0.33, 0.12, 0.43 | 0.45, 0.40, 0.10, 0.35 |
| own / tt_self (H2 needs > 0.46) | 0.19, 0.33, 0.12, 0.41 | 0.34, 0.39, 0.10, 0.35 |
| pure-pair, own: next, head 3 | 0.361, 0.433 | 0.481, 0.437 |
| largest other table's arm / own (P2' needs <= 0.25) | next 0.05, copy 0.49 (topic), topic 0.62 (copy), skip2 0.09 | next 0.02, copy 0.55, topic 1.42 (skip3), skip3 0.02 (shuffled 0.06) |
| head 3 attention at d = 1, 2 (peak) | 0.525, 0.133 (1) | 0.185, 0.393 (2) |
| head 3 own arm vs the other skip table's arm | 0.0367 vs 0.0008 | 0.0269 vs 0.0010 |

On the tests below, seed 0 would give H1 NOT SUPPORTED (P1' fails for every head), H2 NOT STRONGER for every head, and
H3 SUPPORTED (both (a) and (b)). Changing head 3 also moved head 0 (next: more attention at d = 1, pure-pair 0.36 ->
0.48).

## Deviations and clarifications (all before any confirmatory output)

- Hardware (decision 12): development on the laptop GPU; the eight confirmatory runs on one hardware. R2 binds on the
  confirmatory hardware (clarified before any study A output).
- "Another table" in P2' and P3' means the tables of the model's other heads; the alt_ arm (the skip table no head of
  the model has) is used only in H3b.
- A head type inactive (R4) in 2 or more seeds is excluded from H1 and H2 and reported; in the remaining seeds an
  inactive head counts as not meeting the tests. If no head type is active, H1 is NOT SUPPORTED.
- P3' applies to the heads whose tables look locally (next, skip2 in base models); for copy and topic it counts as met.
- The tests below are unchanged from the DRAFT although the development results are known.
- "The joint fit with own and other usefulness" (reported separately) is computed only if time allows (exploratory).

## Question

In a normal model a head's attention and its vote form together, so the law could only be shown as mutual
consistency. Here each head's vote is a fixed, named corpus table and only attention is learned. (1) What attention
does training produce? (2) Does the law, now computable from corpus counts alone, describe it, and is it
head-specific? (3) If one head's table is changed, does that head's attention change as the new table's usefulness
predicts (a causal test)? The human expects no readable language model from this; the study is about how attention
forms. Cost in prediction quality is reported, not tested.

**Why the answer is not given in advance.** At initialisation (attention near uniform, the model near the bigram),
the expected gradient on head h's score for key j at query i is proportional to beta_h times the usefulness of
T_h(t_j) relative to the model's current prediction, centred over keys: the law with the bigram as baseline is the
first-order direction at the start. During training the baseline becomes the model's own prediction, which includes
the other heads' votes (explaining away); attention is limited to a rank-32 bilinear form of a 128-dimensional token
code shared by all heads, plus positions, under a softmax over all keys; and the vote strengths beta co-adapt. The
study measures how much of the trained attention the start-of-training law still describes, and what replaces it.

## Models

    logits_i = beta_follow * log P1(. | t_i) + sum_h beta_h * sum_{j <= i} a^h_ij * T_h(t_j, .)

- a^h: causal softmax of q.k / sqrt(32), q and k from LayerNorm(token embedding + learned position embedding) with
  W_Q, W_K per head (as in TinyLM's attention). The embeddings are used only for attention.
- log P1: FOLLOW's bigram rule (program/corpus.py). T_h: one fixed V x V table per head, distance-independent, row-centred
  under unigram weights (free) and scaled to unigram-weighted RMS 1:
  `next` = PMI_1 (what follows s), `copy` = 1[v = s], `topic` = topic PMI (v 9..24 tokens after s),
  `skip2` = PMI_2, `skip3` = PMI_3. All counted on the TRAIN split (dev and test excluded).
- Learned: token embeddings (256,000), position embeddings (32,768), LayerNorm gain (128), W_Q and W_K (32,768),
  beta_follow (initialised 1) and beta_h (initialised 0.1): about 0.32M parameters (attn models: 0.35M).
- Variants: **base** = heads (next, copy, topic, skip2); **shift** = (next, copy, topic, skip3) (head 3 changed).
- Training as configs/frozen.yaml v3: one epoch (527,482,880 tokens), batch 64 x 256, AdamW (0.9, 0.95), 1% warmup,
  cosine to 10%, weight decay 0.1 on W_Q and W_K only, gradient clip 1.0, no dropout, fp32; peak LR by R1.
- Hardware (decision 12): each comparison within one kind of hardware. The confirmatory runs (base and shift, seeds
  1-4) all on the cloud instance (NVIDIA L4, CUDA, fp32, TF32 off as PyTorch's default) if available, otherwise all on
  the laptop GPU (MPS); development runs (R1 and seed 0) may run on the laptop GPU (they only fix rule values and
  analysis details). The cost comparison (M0) is reported
  against attn_s{s} (laptop) and against study B's lambda = 0 runs of the same seed if they ran on the same hardware.
  Seed s uses the batch order of attn_s{s}; the initialisation differs (different parameters).
- Runs: R1 LR check (base, seed 0, 50M tokens, three LRs); base and shift seed 0 (development); base and shift seeds
  1-4 (confirmatory). Ten one-epoch runs in all.

## Decision rules (development only; values in the table above)

- R1: peak LR = the value in {1e-3, 2e-3, 4e-3} with the lowest dev_full CE after 50M tokens (base, seed 0).
- R2: one epoch per model. If a step on the confirmatory hardware (median of steps 50-150) takes more than 0.25 s, apply
  R3; if still more than 0.25 s, stop and ask the human. Development runs on the laptop GPU are not bound by R2 (they
  run one epoch whatever their speed; operationally they are left to the instance if a laptop step exceeds 0.6 s).
- R3: fp16 autocast for the vote products, only under R2, and only if the loss on 20 fixed batches differs from fp32
  by less than 0.2% (relative).
- R4: a head is **active** if removing its vote (beta_h = 0 at evaluation, attention untouched) raises dev_fast CE by at
  least 0.01 nats. H1 applies to active heads; inactive heads are reported.
- R5: development (seed 0) may fix only these analysis details: the long-range band (default distances 8-63, first
  100M tokens), the usefulness shrinkage (default pseudo-count 20, as in goal 1), and the number of fit and evaluation
  windows (default 32 each). Everything else is fixed by this DRAFT.

## Measurements (per model; windows and conventions as in goal 1: fit on train_windows(32, 2024), output test on
train_windows(32, 2025), pure-pair score at query positions 16, 48, 96, 160, 224)

- **M0 cost:** dev_full CE, paired against attn_s{s} (2.491-2.540); also against the bigram (3.516 dev_full) and program
  v1.3 (dev_fast CE 2.511 against the model's dev_fast CE).
- **M1 vote strengths:** beta at the end and over the checkpoints; the vote-removal shift of each head (R4).
- **M2 usefulness (model-free; computed once from the TRAIN stream after this DRAFT is committed, before any model is
  trained):** for every table T and distance d = 1..7,
  U_T,d(k, q) = (S - N B) / (N + 20), S = sum over positions with k sitting d tokens before q of T(k, next token),
  B = sum_v P1(v | q) T(k, v), N the count (whole stream, as analysis/look_write/uov_tables.py); the same for the
  band d = 8..63 summed (first 100M tokens); and the token-shuffled control (rows of T permuted with default_rng(0)).
  Distance profile: mean usefulness per distance d = 0..7 over all pairs, (sum S - sum N B) / sum N.
  A head's usefulness in logit units is beta_h * U.
- **M3 law fit (as analysis/look_write/fit_arms.py):** the base rule (9 distance buckets, self x query shape, key shape)
  fitted to each head's own pattern; arms add pair features: own usefulness (one coefficient for d = 1, one for
  d = 2..7, one for the band 8..63); each other table's usefulness (same three); the shuffled own table; the exact
  token-token term TT_h(q, k) (token codes with the mean position embedding, one coefficient; the ceiling: at keys
  before the query, as goal 1's pre-registration; `tt_self`: at every key including the current token, as goal 1's
  addendum).
  Output test: substitute the fitted pattern for head h (everything else exact) and measure KL(model || substituted);
  gain of an arm = KL with the base rule's pattern - KL with the arm's pattern.
- **M4 pure-pair correlation at d = 1:** the model's exact score for key k one before query q minus q's self score,
  averaged over the five positions, double-centred and weighted by N_1 over pairs with N_1 >= 20, correlated with each
  table's U at d = 1 (as goal 1).
- **M5 distance profile of attention:** mean over queries i >= 7 of a^h_{i, i-d}, d = 1..7.
- **M6 coupling:** the fitted coefficients (attention score per logit of beta_h * U), against goal 1's 0.24.

## Hypotheses and tests (confirmatory seeds 1-4)

**H1: the law describes the learned attention when votes are fixed** (base models; per active head):
- P1': own-usefulness gain >= 0.5 x the ceiling's gain. If the ceiling's gain is <= 0.01 nats the ratio is undefined
  and P1' counts as not met.
- P2': every other table's arm and the shuffled arm gain <= 0.25 x the own arm's gain.
- P3' (heads `next` and `skip2`, which look locally): pure-pair correlation with own U >= 0.25 and above every other
  table's.
A head type follows the law if P1'-P3' all hold in at least 3 of 4 seeds. **SUPPORTED** if every active head type
follows it; **PARTIAL** if at least one does; **NOT SUPPORTED** if none does.

**H3: changing a head's table changes where it looks, as the new table's usefulness predicts** (head 3):
- (a) Distance: the peak of head 3's attention over d = 1..7 (M5) is at the distance where its table's mean usefulness
  (M2 profile) peaks over d = 1..7: in base models for skip2, in shift models for skip3; at least 3 of 4 seeds of
  each variant. (The predicted distances are fixed when M2 is computed, before any model is trained. If head 3's
  beta ends negative, its usefulness in logit units is reversed and the predicted peak is the minimum over d = 1..7.)
- (b) Crossover: in shift models head 3's arm with skip3 usefulness gains more than its arm with skip2 usefulness; in
  base models the reverse; at least 3 of 4 seeds of each variant.
**SUPPORTED** if (a) and (b) hold; **PARTIAL** if one holds; **NOT SUPPORTED** if neither.

**H2: the law is stronger when votes are fixed than in normal models** (base models; per active head type): the
own-usefulness gain divided by the `tt_self` gain exceeds 0.46 (the top of goal 1's range, 26-46%) in at least 3 of 4
seeds. Reported per head type as STRONGER or NOT STRONGER; goal 1's pure-pair correlations (0.34-0.44) are reported next
to the new ones without a threshold.

## Reported separately (no verdict weight)

Cost (M0, mean and SE of the paired differences); vote strengths and their trajectories; coupling against goal 1;
heads 0-2 in shift vs base models (they have the same tables: does changing head 3 move them?); the joint fit with own
and other usefulness (the unique part of own usefulness, since tables are correlated).

## Pipeline checks before any confirmatory analysis

- The law-fit code, run on attn_s1 with its own OV tables, reproduces goal 1's own-arm gains (within 0.002 nats).
- Synthetic check: when a head's pattern is replaced by hand with softmax(base rule + c x own usefulness), the own arm
  recovers c within 10% and its gain is at least 0.9 of the base rule's KL (it describes that pattern almost exactly;
  the token-token ceiling does not apply to a hand-made pattern).

## Recorded expectations (no verdict weight)

DRAFT: dev CE between the attn models' and the bigram's, nearer the attn models (about 2.6-2.7); skip2 peaks at
distance 1 and skip3 at distance 2; next and skip2 follow the law more closely than in normal models; copy and topic
look far and their law test relies on the band feature. After development: H3 SUPPORTED, H1 NOT SUPPORTED, H2 NOT
STRONGER are the most likely outcomes.

## Exploratory

Growth over training (checkpoints: when the own-usefulness correlation appears relative to beta_h); usefulness relative
to the model's own prediction instead of the bigram (explaining away: does the law describe more of the attention with
the right baseline?); a readable description (tables, fitted rules,
the law) and its KL to the model; further table changes. No use of the test split.
