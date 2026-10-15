# Pre-registration: does "where a head looks comes from what it writes" hold in deeper models? (DRAFT)

DRAFT. Placeholders marked {..} are filled by the decision rules R1-R3 of logs/deep_study.md (committed before any
calibration output). This file is committed before any arm is fitted on the confirmatory seeds.

## What has been seen before writing this

- One-layer models (attn_s0..s4): the pre-registered replication (logs/look_follows_write_results.md) and the addendum
  (logs/look_follows_write_addendum_results.md). Head-specificity held in all 8 confirmatory heads. Against the full
  token-token attention term (`tt_self`), own-vote usefulness reaches about a third of the output gain (0.26-0.46).
- Calibration on attn_s1..s4 with this study's pipeline (200M-token samples): {readout comparison, own/oracle ratios,
  coefficient medians}.
- Development seeds attn2L_s0 and attn4L_s0: head classification, selftests and dev loss only. No arm fitted, no
  outcome computed.

## Models

Attention-only, 2 and 4 blocks (configs/side_deep_attn.yaml), otherwise frozen v3. Confirmatory: attn2L_s1..s4 and
attn4L_s1..s4 (seeds 1-4, one epoch each). Seed 0 of each depth is the development seed.

## Head classes and families (analysis/look_write_deep/classify.py, fixed rules)

- LOCAL: pooled mean attention distance to earlier tokens <= 10.
- PREV: mean attention on the previous token >= 0.5.
- INDUCTION: score >= 0.3 on 32 synthetic repeated random sequences.
- Confirmatory head = LOCAL, not PREV, not INDUCTION, earlier-token mass >= 0.15.
- F1 = layer 0 of every model; F2 = the last layer; F3 = layers 1-2 of the 4-block models.

## Definitions (analysis/look_write_deep/)

- Bigram-stage usefulness (BGU) of head h's write from key j for query i: u = o_h(j) . g_i, where g_i is the gradient
  of F = sum_i <logits_i, e_{y_i} - P1(. | t_i)> w.r.t. the residual after the head's layer. Aggregated per (k, q, d),
  d = 1..7, over a 200M-token TRAIN sample (fit and eval windows excluded): sum u / (N + 20).
- Primary readout: {LTOTAL: later blocks exact, final LayerNorm linearised at the typical scale / TOTAL: exact
  gradient}. DV (direct path only) is the nested comparison for D4.
- Base attention rule: 9 distance buckets, self x query shape, key shape; fitted by pattern cross-entropy on
  train_windows(32, 2024). Pair arms add c1 * f at d = 1 and c2 * f at d = 2..7.
- Output test: the arm's pattern substituted for the head's pattern, later blocks recomputed, KL vs the model on
  train_windows(64, 2025); paired SEs over windows.
- Ceiling: the oracle table OQC = the head's own mean score s_ij - s_ii per (k, q, d), shrunk by N / (N + 20).
- Controls: every other head's BGU table (primary readout), and a value-shuffled control (token-level write of a
  permuted token). Other heads whose BGU table (d = 1) correlates > 0.5 with the head's own are listed as
  near-duplicates and are not used as controls.
- Pure-pair target: the oracle table at d = 1, double-centred over pairs with sample N >= 20 (weights N).

## Predictions (per confirmatory head)

- D1: if the oracle gain >= max(0.01 nats, 3 paired SE): own gain >= theta x oracle gain, theta = {0.5 or rule R2};
  otherwise own gain >= 0.15 x base KL.
- D2: max(non-duplicate other heads' gains, shuffled gain) <= 0.25 x own gain (and own gain > 0).
- D3: own pure-pair correlation >= 0.25 and above every non-duplicate other head's.
- D4 (F1, F3): primary gain >= DV gain - 2 SE for every head, and primary >= DV + 2 SE for at least 75% of the heads
  whose primary and DV tables correlate below 0.5 at d = 1.
- D5 (reported separately): own coefficients c1, c2 within a factor 1.5 of the one-layer medians {c1 = .., c2 = ..}.

Verdict per family: SUPPORTED if D1-D3 hold for at least 80% of its confirmatory heads; REFUTED if D1 fails for at least
50%; otherwise MIXED; INCONCLUSIVE with fewer than 4 heads. Headline ("generalises to deeper heads") = F1 and F2 both
SUPPORTED. A claim of direction ("comes from") additionally needs the echo test (addendum D on the one-layer
checkpoints, and the same on the deep checkpoints, exploratory) to support it.

## Exploratory (no verdict weight)

PREV, INDUCTION and far-looking heads; SAME readout; freeze-later split; echo decomposition (MGU, CGU arms); the
development seeds.

## Per-token code test (analysis/look_write_deep/code_test.py, formula fixed)

Held-out heads: the 8 confirmatory heads of attn_s1..s4 and the confirmatory layer-0 heads of the deep seeds 1-4.
attn_s0 was used for development: there, the top-32 corpus subspace held 15-22% of a local head's vote energy, the
model's own embedding subspace 39-67%, C_pmi beat C_res, and the far-looking head scored highest.
- K1a: E32(C_res) above the 95th percentile of random 32-dim subspaces of the corpus top 128, for at least 80% of
  held-out heads. (Expected to hold.)
- K1b: E32(C_res) >= 1.5 x E32 of the model's embedding subspace, for at least 80%. (Expected to fail.)
- K2: R2_diag(32) >= 0.8 x R2_diag(128).
- K3: LOCAL heads have higher E32 than every RECALL head of the same model. (Expected to fail.)
- K4: C_res gives higher E32 than C_pmi for most heads. (Expected to fail.)
If K1b fails, the conclusion is that the heads' vote codes follow the model's embedding geometry rather than the
best low-rank compression of these corpus statistics.
