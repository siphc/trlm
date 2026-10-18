# Where an attention head looks follows from what it writes — in heads that write to the output

*Pre-draft, 2026-10-17. Every number is from a committed results file (sources in brackets); nothing here is new
analysis. Sections marked TODO list what a real draft still needs.*

## Abstract

Why an attention head attends where it does is a central open problem in mechanistic interpretability: its
query-key (QK) circuit is usually described, not explained. We test a specific answer in small transformers trained
on TinyStories: the pair-specific part of a head's attention score is set by how useful the head's own vote (its
value-output write) from the attended token would be for predicting the next token, beyond what the current token
alone predicts ("own-vote usefulness"). The hypothesis was found on one model and then tested under pre-registration.
In one-layer attention-only models (four confirmatory seeds), the law holds for every local head and is head-specific:
the attention pattern lines up with the usefulness of the head's own vote (pure-pair correlation 0.34-0.44) and not
with another head's vote or a shuffled vote (at most 0.08). It accounts for about a third (26-46%) of what the exact
token-token attention term buys in output. Over training, attention and vote co-adapt: neither forms first, but
usefulness predicts which token pairs come to be strongly attended. In 2- and 4-block models the law holds for
last-layer heads (9 of 16 meet all pre-registered criteria; 15 of 16 the main one) and is refuted for heads whose
writes are read by later layers (first and middle layers), even when usefulness is propagated through the later
layers. The coupling constant is not universal (about 0.25 in one layer, 0.65-0.8 in the last layer of deeper
models). The law is a property of heads that write to the output.

## 1. Introduction

A one-layer attention-only transformer computes its logits as a sum: the direct path (the current token's own
prediction) plus, for each head, an attention-weighted average of what the head writes for each earlier token
(Elhage et al. 2021). The value-output (OV) circuit, what a head writes, often has a readable description (copying,
copy suppression, bigram-like votes). The query-key (QK) circuit, where a head looks, is harder; recent work calls
attention pattern formation the main remaining problem for attention and reports only weak coupling between QK and OV
in large models (Anthropic 2025a, b; TODO: verify wording against the full texts).

Training-dynamics theory suggests a link: gradient descent increases a key's attention score where the head's value
vector for that key points along the prediction error (Tian et al. 2023; Tarzanagh et al. 2023; Bietti et al. 2023,
who find that output associations form first and attention then focuses on them). If this holds in trained models,
then where a head looks should be predictable from what it writes plus corpus statistics.

We make this measurable as own-vote usefulness: for a head h, a key token k at distance d before a query token q, how
much h's vote from k (its OV row) would help predict the token after q, beyond the prediction of q alone. We then test,
with pre-registered predictions and controls, whether usefulness predicts the pair-specific part of each head's
attention.

**Contributions.**
1. A head-specific law in one-layer models: attention follows the usefulness of the head's own vote, not another
   head's, in every local head of four held-out seeds; it explains about a third of the exact token-token attention
   term's output effect.
2. Direction over training: attention and vote co-adapt; usefulness predicts which pairs become strongly attended
   even after matching on current attention.
3. Scope in deeper models: the law holds for output-layer heads and fails for heads whose writes feed later layers,
   even with usefulness measured through those layers.
4. A negative result on corpus codes: heads' vote codes follow the model's own embedding geometry, not the best
   low-rank compression of corpus statistics.
5. A use: the law, computed from corpus counts, became part of a readable replacement program for one model.

## 2. Setting

**Models.** Decoder-only transformers trained on TinyStories (Eldan & Li 2023) with a 2,000-token BPE vocabulary,
context 256, d_model 128, 4 heads of size 32 per layer, learned absolute positions, pre-LayerNorm, tied embeddings, no
MLP, one epoch, seeds 0-4 (configs/frozen.yaml). One-layer models attn_s0..s4 (dev loss 2.49-2.54). Deeper models
(side study, same settings): 2 blocks attn2L_s0..s4 (dev loss 2.27-2.33) and 4 blocks attn4L_s0..s4 (2.13-2.15)
[logs/deep_study.md]. None of the deeper models forms induction heads: on a repeated random sequence, the loss on the second copy
is 13.4 nats (2 blocks) and 15.3 (4 blocks) against 14.8 and 15.3 on the first copy; an induction circuit would bring it
to a few nats [logs/deep_study.md].

**Seeds.** Seed 0 is the development seed at every depth; seeds 1-4 are confirmatory.

**Head classes** (fixed rules). LOCAL: mean attention distance to earlier tokens at most 10 tokens; otherwise RECALL
(far-looking). Deep study only: PREV (previous-token attention >= 0.5) and INDUCTION heads are excluded; a
confirmatory head also needs earlier-token attention mass >= 0.15.

## 3. Method

**Own-vote usefulness, one layer (UOV).**
UOV_h,d(k, q) = E[OV_h(k, next) | k sits d tokens before q] - E_{v ~ P1(.|q)}[OV_h(k, v)], where OV_h(k, v) is the
logit head h adds to v when attending fully to k (in logit units, final LayerNorm at its typical scale), the first
expectation runs over all corpus positions with k at distance d before q (the next token is the one after q), and
P1(.|q) is a corpus bigram ("what q alone predicts"). Shrinkage (S - N B)/(N + 20) [logs/prereg_look_follows_write.md].

**Bigram-stage usefulness, deeper models (BGU).** u = o_h(j) . g_i, with o_h(j) the head's write from key j and g_i the
gradient of sum_i <logits_i, e_{y_i} - P1(.|t_i)> with respect to the residual stream after the head's layer, averaged
per (k, q, d) over a 200M-token sample. Readouts: DV (direct path only), LTOTAL (later blocks exact, final LayerNorm
linearised at its typical scale; equals DV in one layer), TOTAL (exact gradient). LTOTAL was fixed as primary on the
one-layer calibration before any confirmatory fit (median own gain 0.104 vs 0.049 for TOTAL)
[logs/prereg_deep_look_write.md].

**Tests (identical for every head).**
- *Base rule:* a generic log-linear attention rule (distance buckets, self x query shape, key shape), fitted to the
  head's own attention pattern.
- *Arms:* add one pair feature (coefficients c1 at d = 1, c2 at d = 2..7): the head's own usefulness; each other
  head's usefulness; a token-shuffled control; a ceiling (one layer: the exact token-token QK term; deep: an oracle
  table of the head's own mean pair scores).
- *Output test:* substitute the arm's pattern into the exact model (everything else exact), KL to the model on
  held-out windows; gain = KL(base) - KL(arm), paired SEs over windows.
- *Pattern test:* "pure-pair" correlation at d = 1 between the feature and the head's own pair scores, both
  double-centred (query and key main effects removed), weighted by pair counts.

**Discipline.** Hypothesis found on seed 0 (discovery round 2); one-layer pre-registration before seeds 1-4 were
analysed; two addenda written before their analyses; deep-study decision rules (readout, threshold, reference
constants) fixed before the calibration output, and the FINAL deep pre-registration committed before any confirmatory
fit [logs/prereg_*.md; commits 9c149a1, 8ab4a9f].

## 4. Results

### 4.1 Discovery (seed 0; motivating, not evidence)

In attn_s0, a two-scalar usefulness term added to each local head's attention rule beat the exact token-token QK term in
output KL (head 0: 0.266 vs 0.318; head 2: 0.222 vs 0.264; head 3: 0.229 vs 0.226), with a coefficient of about 0.3 per
nat for every head and distance; another head's vote predicted nothing [logs/mech_v2_investigations.md].

### 4.2 One-layer replication (pre-registered; seeds 1-4, 8 LOCAL heads)

| head | own gain | exact TT gain | own / TT | max control / own | own corr. (max other) |
|---|---|---|---|---|---|
| s1/h3 | +0.092 | -0.052 | (TT negative) | -0.01 | 0.404 (0.057) |
| s2/h2 | +0.115 | +0.055 | 2.09 | 0.04 | 0.368 (0.015) |
| s3/h0 | +0.112 | +0.057 | 1.97 | 0.01 | 0.335 (-0.003) |
| s3/h1 | +0.064 | +0.168 | **0.38** | 0.10 | 0.436 (0.082) |
| s3/h3 | +0.089 | +0.117 | 0.76 | 0.01 | 0.378 (0.018) |
| s4/h0 | +0.176 | +0.142 | 1.25 | 0.02 | 0.371 (-0.025) |
| s4/h2 | +0.102 | +0.017 | 5.98 | 0.03 | 0.415 (0.034) |
| s4/h3 | +0.071 | +0.084 | 0.84 | 0.05 | 0.436 (0.059) |

[analysis/look_write/RESULTS.md; gains in nats; own-gain SEs 0.002-0.008]

- **Verdict: SUPPORTED on the literal reading (7 of 8 meet P1-P3; bar 80%), MIXED on the strict reading (6 of 8).** The
  difference is s1/h3, whose pre-registered ceiling makes the output worse (-0.052 +- 0.011), so its ratio is
  meaningless.
- **Head-specificity is robust:** own correlation 0.34-0.44 in every head; other heads at most 0.08; controls at most
  0.10 of the own gain.
- **A roughly shared constant:** medians c1 0.233, c2 0.251 attention-score units per logit of usefulness; 7 of 8 heads
  within a factor 1.5.

### 4.3 How much of the pair term? (addendum A)

With the exact token-token term also applied at the self position (the strictest ceiling), the ceiling gains become
0.20-0.39 nats per confirmatory head and positive everywhere, and own usefulness reaches **0.26-0.46 of it (A2 predicted >= 0.5: 0
of 8)**. The gap is the joint, token-specific weighting of earlier keys relative to the current token; usefulness
explains about a third of it [logs/look_follows_write_addendum_results.md].

### 4.4 Far-looking heads (addendum C)

Usefulness aggregated over distances 8-63 beats the d = 1..7 version exactly for the truly far-looking heads (mean
distance 25-31 tokens; 4 of 8 RECALL heads), with controls at most 0.25 of the own gain in 7 of 8. Gains are smaller
(0.05-0.07 nats) than the full token-token term's (0.15-0.21).

### 4.5 Direction over training (addendum D and addendum 2)

Concern: in a well-trained, calibrated model, usefulness could mirror attention without causing it (the head's own
attention is part of what the model predicts). Tested on training checkpoints of attn_s1..s4:

| checkpoint (step) | LR / peak | dev loss | r median (min) |
|---|---|---|---|
| 229 | 0.78 | 3.56 | 0.05 (0.01) |
| 718 | 1.00 | 3.56 | 0.23 (0.19) |
| 2246 | 0.99 | 2.88 | 0.41 (0.37) |
| 7032 | 0.90 | 2.67 | 0.46 (0.44) |
| 22009 | 0.31 | 2.57 | 0.45 (0.39) |
| final | 0.10 | 2.53 | 0.43 (0.36) |

- The relation is near full strength early (step 2246: 95% of its final value, at full learning rate, 0.35 nats above
  the final loss).
- **Cross-lagged test fails:** attention change vs earlier usefulness +0.002; usefulness change vs earlier attention
  +0.090. Late in training the vote adapts to where the head already looks.
- **Growth test:** votes do not settle before attention (G1: 5 of 8, predicted >= 6). But token pairs that newly enter
  a head's top 10% of attention had higher usefulness at the previous checkpoint in all 8 heads (+0.35 to +0.54),
  and still after matching on current attention (+0.17 to +0.28, p = 0.0005 each). "Recursive generalisation" of the
  key set (G3) does not survive the same control.

**Reading:** co-adaptation. Attention and vote form together; usefulness leads at the top of the attention
distribution; across the bulk of pairs and late in training, the vote follows the attention. A strict "comes from"
is not supported; "is predictable from the vote and the data at every stage after step 229" is.

### 4.6 Deeper models (pre-registered; 2 and 4 blocks, seeds 1-4)

| family | heads | D1 | D2 | D3 | D1-D3 | verdict | median own gain | own / oracle | pair corr. | max control / own |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 last layer | 16 | 15 | 14 | 11 | 9 | MIXED | 0.065 | 0.43 | 0.31 | 0.06 |
| F1 layer 0 | 11 | 5 | 3 | 2 | 1 | REFUTED | 0.007 | 0.15 | 0.19 | 0.45 |
| F3 middle layers | 22 | 8 | 15 | 7 | 4 | REFUTED | 0.012 | 0.15 | 0.23 | 0.19 |
| (one-layer calibration) | 8 | | | | | | 0.104 | 0.37 | 0.40 | 0.02 |

[logs/deep_look_write_results.md; D1: own gain >= 0.20 x oracle gain; D2: controls <= 0.25 x own; D3: own pure-pair
correlation >= 0.25 and above all other heads]

- **Headline fails:** the law does not generalise to deeper heads as a whole.
- **Output-layer heads follow it:** last layer of 2-block models 8 of 11 heads meet D1-D3; of 4-block models 1 of 5.
- **Heads that write for later layers do not**, even at the same depth as one-layer heads (layer 0).
- **Propagating usefulness through later layers does not help (D4 fails):** LTOTAL predicts attention no better than
  the direct path for first- and middle-layer heads.
- **The constant is not universal (D5 fails):** last-layer coefficients c1 0.65, c2 0.79 against 0.24 / 0.27 in one
  layer.

### 4.7 Are vote codes corpus statistics? (code test, held-out heads)

Vote-table energy in the top-32 subspace of a corpus usefulness statistic: median 0.197, versus 0.038 for random
32-dimensional subspaces (K1a holds 19 of 19) and 0.565 for the model's own embedding subspace (K1b fails 0 of 19, as
predicted). Heads' vote codes lean toward the corpus statistic's main directions but follow the model's embedding
geometry.

### 4.8 Capacity (exploratory)

Single-head earlier-token ablations: in one-layer models, 33-83% of head pairs have anti-correlated effects (mean
correlation -0.08 to -0.10); in 2-block models 0-11% (+0.04 to +0.05); in 4-block models none (+0.05 to +0.06), even
for heads in the same layer. One-layer heads partly cancel each other; deeper models' heads do not.

### 4.9 Verification

An independent agent re-implemented the verdict logic from the pre-registration text alone (all 49 deep heads match
field by field), re-ran one model's fits (8,472 values bit-identical), and audited the code numerically (float64
re-implementation within 1.8e-6; finite differences within 1e-7). Resampling the evaluation windows 2,000 times
reproduces the family verdicts in 99.3-100% of draws [logs/deep_look_write_results.md, Verification].

## 5. Use: a readable replacement program

The law, computed from corpus counts and the program's own votes instead of the model's weights, was added to the
local readers of a readable NumPy program that reproduces attn_s0 from corpus statistics and a few hundred named
scalars. Each reader's attention gains 0.35-0.55 per nat of usefulness of its own vote: +0.0070 +- 0.0005 nats on fresh
windows (dev KL 0.578 -> 0.571), and an older hand-made attention term (key informativeness) halved its weight, absorbed
by the law [logs/look_follows_write_results.md]. The gain is small next to the exact model's (0.06-0.18 nats per head)
because the program's votes only partly match the heads' votes; the law moves the problem of explaining attention onto
the problem of explaining votes.

## 6. Discussion

**What the law is.** For a head whose write goes straight to the output, the pair-specific part of its attention
tracks how much that write would reduce the next-token error relative to a bigram, specifically for its own write. This
is what first-order training dynamics predict, and it holds in trained models; it is the empirical counterpart of
"attention selects tokens whose values help". The co-adaptation result sharpens the interpretation: the law describes a
consistency between QK and OV maintained during training, with usefulness leading only for the strongly attended pairs.

**Why it fails for earlier heads.** A head whose output is read by later heads is not trained to reduce the next-token
error directly with its own write; its useful role is to supply inputs to later heads (composition). Bigram-stage
usefulness propagated through the later layers (LTOTAL) captures what the head's write does to the output given fixed
later attention, but not what later heads need from it. The study's models have no induction heads, so composition here
is of a quieter kind; what sets those heads' attention remains open.

**Constants.** The coefficient differs by depth (0.25 vs 0.65-0.8). It may reflect how strongly the final LayerNorm and
the residual stream's size scale usefulness in different architectures; not tested.

**Relation to other work.**
- Attention-dynamics theory (Tian et al. 2023; Tarzanagh et al. 2023; Bietti et al. 2023; Nichani et al. 2024): we test
  their prediction in trained language models; the strict order "outputs first, then attention" (Bietti et al.) is not
  what we see (G1 fails), but usefulness-led acquisition at the top of the attention distribution is.
- Interpretability of attention (Anthropic 2025a, b; Hayes, Li & Andreas 2026, who write programs reproducing attention
  patterns): our law explains part of the QK circuit from the OV circuit and data. A side observation: fitting an
  attention rule more closely to a head's pattern can make the output worse, so pattern similarity alone is a weak test
  of a head explanation [logs/mech_v2_investigations.md].
- Cammarata (2026) argues that established routes capture residual error first; the usefulness measure is a
  quantitative version of that idea applied to attention. The capacity result is in the direction his essay predicts
  (with fewer units, broader and more shared work), exploratory.

## 7. Limitations

- Tiny models (d_model 128, 4 heads per layer), one corpus (TinyStories), one epoch, attention-only (the MLP model's
  heads act through the MLP, and the direct-path test there is not meaningful).
- Usefulness is measured with the model's own write (from the weights) and against a corpus bigram baseline; it is a
  relation between two parts of the model, not a weight-free explanation.
- Pair tables cannot express context-dependent queries and keys; failures in first and middle layers may partly be
  the instrument's (stated in the pre-registration as a possible null interpretation).
- The one-layer verdict rests on one head (literal 7/8 vs strict 6/8); the first-layer refutation in deep models rests
  on one head too (6 failures needed, 6 found), though the failing heads are far from the threshold and resampling
  gives the same verdict 99.3% of the time.
- No induction heads, so in-context heads were not tested.

## 8. TODO for a draft

- Figures: (1) usefulness vs exact pair score scatter for one head; (2) r(c) over training with the calibration index;
  (3) family bars (own / oracle, controls) for 1-, 2-, 4-block models.
- Replication outside our models: a public 2-layer TinyStories model or GPT-2 small (last-layer heads first).
- Check every related-work claim against the full papers (the survey used abstracts; e.g. Anthropic's wording).
- Decide the headline: "a law for output-writing heads" vs "co-adaptation of QK and OV".
- The coupling constant: test whether it follows the final LayerNorm scale or residual size.

## References

- Anthropic (2025a). Progress on Attention. https://transformer-circuits.pub/2025/attention-update/index.html
- Anthropic (2025b). Tracing Attention Computation Through Feature Interactions. https://transformer-circuits.pub/2025/attention-qk/
- Bietti, A. et al. (2023). Birth of a Transformer: A Memory Viewpoint. arXiv:2306.00802
- Cammarata, N. (2026). Neural Networks Are Already Clean. Goodfire. https://already-clean.goodfire.com/
- Eldan, R. & Li, Y. (2023). TinyStories. arXiv:2305.07759
- Elhage, N. et al. (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub
- Hayes, Li & Andreas (2026). Explaining Attention with Program Synthesis. arXiv:2606.19317
- Nichani, E. et al. (2024). How Transformers Learn Causal Structure with Gradient Descent. arXiv:2402.14735
- Tarzanagh, D. A. et al. (2023). Transformers as Support Vector Machines (max-margin token selection). arXiv:2308.16898
- Tian, Y. et al. (2023). Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer. NeurIPS 2023

(Reference list from an earlier search that used abstracts; to verify before a draft.)

## Appendix: where everything is

| part | pre-registration | results | code |
|---|---|---|---|
| one-layer replication | logs/prereg_look_follows_write.md | logs/look_follows_write_results.md, analysis/look_write/RESULTS.md | analysis/look_write/ |
| addenda A-D | logs/prereg_look_follows_write_addendum.md | logs/look_follows_write_addendum_results.md | analysis/look_write/, analysis/look_write_deep/echo_traj.py |
| growth test | logs/prereg_look_follows_write_addendum2.md | same file, "Addendum 2" | analysis/look_write_deep/growth.py |
| deep study | logs/prereg_deep_look_write.md | logs/deep_look_write_results.md | analysis/look_write_deep/ |
| study log | | logs/deep_study.md | |
