# Results: does "where a head looks comes from what it writes" hold in deeper models? (2026-10-17)

Pre-registration: logs/prereg_deep_look_write.md (FINAL, commits 9c149a1 and 8ab4a9f, before any confirmatory fit).
Rules: primary readout LTOTAL, theta 0.20, D5 references c1 0.243 and c2 0.271 (analysis/out/look_write_deep/rules.json).
Models: attention-only, 2 and 4 blocks, seeds 1-4 (confirmatory). Outputs: analysis/out/look_write_deep/summary_ltotal.json,
<model>/arms_ltotal.json, code_test.json, <model>/capacity.json. Independent verification: see the last section.

## Verdict

**The headline prediction fails: the law does not generalise to deeper heads as a whole.** It holds well for the heads
whose writes go straight to the output, the last layer (MIXED by the strict rule, 9 of 16 heads meet D1-D3), and fails
for the heads whose writes are read by later layers (layer 0 and the middle layers: REFUTED).

| family | heads | D1 | D2 | D3 | D1-D3 | verdict | median own gain | own / oracle | pair corr. | max control / own |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 last layer | 16 | 15 | 14 | 11 | 9 | MIXED | 0.065 | 0.43 | 0.31 | 0.06 |
| F1 layer 0 | 11 | 5 | 3 | 2 | 1 | REFUTED | 0.007 | 0.15 | 0.19 | 0.45 |
| F3 middle layers (4 blocks) | 22 | 8 | 15 | 7 | 4 | REFUTED | 0.012 | 0.15 | 0.23 | 0.19 |
| (one-layer calibration) | 8 | | | | | | 0.104 | 0.37 | 0.40 | 0.02 |

- **F2 split by depth:** last layer of the 2-block models 8 of 11 heads meet D1-D3; of the 4-block models 1 of 5.
- **D4 fails** (F1 and F3): reading the write through the later blocks (LTOTAL) predicts attention no better than the
  direct path (DV); for most earlier-layer heads it is slightly worse (F1: 5 of 11 not worse than DV; F3: 7 of 22).
- **D5 fails:** last-layer coefficients are much larger than in one-layer models (medians c1 0.65, c2 0.79 vs 0.24,
  0.27); F1 0.13 / 0.07; F3 0.33 / 0.28. No shared constant across depths.

## What this says

1. **The law is a property of heads that write to the output.** In one-layer models every head writes to the output and
   every local head follows the law. In deeper models the last-layer heads follow it with similar or larger effect
   sizes, and head-specifically (other heads' tables and the shuffled control give 6% of the own gain). Heads at the
   same depth as the one-layer heads (layer 0) but whose writes are mostly read by later layers do not.
2. **Counting what later layers read does not rescue the earlier heads.** Bigram-stage usefulness propagated through the
   later blocks (LTOTAL) predicts their attention no better than their direct effect on the output. Whatever sets
   where an early or middle head looks, it is not the next-token usefulness of its write at the same position, even
   when that usefulness includes the later layers. Their attention is plausibly shaped by what later heads need from
   them (composition), which this measure does not capture.
3. **The coupling constant is not universal.** It is about 0.25 in one-layer models and about 0.65-0.8 in the last
   layer of deeper models.

## Per-token code test (formula fixed in advance; 19 held-out heads: 8 one-layer, 11 deep layer-0)

| prediction | result | expected |
|---|---|---|
| K1a: E32(C_res) above random 32-dim subspaces of the corpus top 128 | 19 of 19, holds | hold |
| K1b: E32(C_res) >= 1.5 x the model's embedding subspace | 0 of 19, fails | fail |
| K2: R2_diag(32) >= 0.8 x R2_diag(128) | 17 of 19, holds | (no expectation) |
| K3: LOCAL heads above every RECALL head of the same model | 1 of 11 models, fails | fail |
| K4: C_res above C_pmi | 0 of 19, fails | fail |

Median E32: corpus top-32 subspace 0.197, random 0.038 (95th percentile), the model's own embedding subspace 0.565.
**Conclusion as pre-stated:** the heads' vote codes follow the model's embedding geometry rather than the best
low-rank compression of these corpus statistics, although they do lean towards the corpus statistic's main directions.

## Capacity comparison (exploratory; all 5 seeds per depth)

Single-head earlier-token ablations; rho = probability-weighted correlation of their effects on the output.

| depth | heads | share of head pairs with rho < -0.1 | mean rho | same-layer mean rho |
|---|---|---|---|---|
| 1 block | 4 | 0.33-0.83 | -0.08 to -0.10 | (all same layer) |
| 2 blocks | 8 | 0.00-0.11 | +0.04 to +0.05 | -0.01 to -0.03 |
| 4 blocks | 16 | 0.00 | +0.05 to +0.06 | -0.01 to +0.01 |

One-layer heads partly cancel each other; in deeper models they do not, even within a layer of the same width (4
heads). Direction as predicted by Cammarata's essay (fewer units: broader, shared work). Exploratory.

## Combined with the one-layer results

- One layer: every local head follows the law, head-specifically; it accounts for about a third of the token-token
  attention term; attention and vote co-adapt during training (usefulness predicts which pairs become strongly
  attended; the vote adapts to the attention late in training).
- Deeper models: the law holds for heads that write to the output (strongly in the 2-block models' last layer, weakly in
  the 4-block models'), not for heads that write for later layers, and its constant depends on depth.
- A head's vote code lives in the model's own embedding geometry, not in a low-rank corpus statistic.

## Limitations

Tiny models (d_model 128, 4 heads per layer), one corpus (TinyStories), one-epoch training; usefulness measured against a
corpus bigram baseline and with the model's own write; token-pair tables cannot express context-dependent queries and
keys, so a failure in F1/F3 may partly reflect the measuring instrument (stated in the pre-registration as a possible
null interpretation). No induction heads formed in these models, so in-context heads were not tested.

## Verification (independent agent, no shared code; scripts in analysis/out/look_write_deep/verify/)

1. **Verdict logic re-implemented from the pre-registration text only** (verify_verdicts.py -> verdicts_verify.json):
   all 49 confirmatory heads match summary_ltotal.json field by field (gains, paired SEs, ratio-evaluability, D1-D5,
   controls, pure-pair and table correlations, c1/c2, near-duplicates); 0 discrepancies; family counts and verdicts
   identical. rules.json recomputed from the calibration arms: same values (median own gain LTOTAL 0.1041 vs TOTAL
   0.0492; median own/oracle ratio 0.3723, theta 0.20; c1 0.2426, c2 0.2710).
2. **Reproduction:** fit_arms_deep re-run on a copy of attn2L_s1 (verify2L_s1/): all 8,472 stored values bit-identical
   to the committed arms_ltotal.json.
3. **Code audit and numerical checks** (pass_bgu, common_deep, fit_arms_deep against the pre-registration): BGU,
   LTOTAL (final LN linearised at scale 0.8330), oracle, shuffled control, window exclusion (36 fit/eval windows that
   would have been in the 200M sample are excluded), substitution and pair features all as stated. A float64
   re-implementation agrees with pass_bgu within 1.8e-6 relative on a 16-window pass; finite differences agree with
   the autograd gradients within about 1e-7; 36 random committed 200M-token slots of attn2L_s1 and attn4L_s3
   recomputed from all their windows match (median relative difference <= 2e-6); head classes recomputed for all 8
   models (no change); L-BFGS converged (max gradient about 1e-8) for three borderline heads.

Corrections and notes from the verification:
- The F3 median own gain is 0.012 (0.01247), not 0.013 as in the first draft of this file (corrected above).
- The pre-registration says F3 has 21 heads; its own fixed classification rules give 22 (recomputed independently).
  No 21-head subset changes the verdict (at least 13 of 21 would still fail D1).
- The D1 floor fallback (0.15 x base KL) was never used: every head's oracle gain is >= 0.0231.
- Only one near-duplicate exists (attn2L_s3 L1H1 ~ L0H3); excluding it changes nothing.
- **Convention that matters:** the pure-pair correlation (D3) double-centres both the oracle and the usefulness table,
  as the one-layer pre-registration specified. Without double-centring the usefulness table, 17 of the 20 heads that
  pass D3 would fall below 0.25. The deep pre-registration inherited this convention without restating it.
- Near-duplicate and "tables differ" correlations are N-weighted. Unweighted, a few flags change, but D4 still fails on
  its "not worse than DV" clause.
- The D3 pair set has 176,589 pairs, not every pair with sample N >= 20: 7,317 pairs (0.08% of the weight) lack a
  table entry because table slots are chosen by expected whole-stream count. Negligible.
- **Margin (not pre-registered):** F1 is REFUTED by one head (6 D1 failures needed, 6 found), but the failing heads'
  own/oracle ratios (0.002-0.147) are far from theta 0.20. Resampling the 64 eval windows 2000 times: F1 REFUTED in
  99.3% of draws, F2 MIXED in 100%, F3 REFUTED in 100%.
