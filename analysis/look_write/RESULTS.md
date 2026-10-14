# Results: "where a head looks comes from what it writes" (pre-registered replication, 2026-10-14)

Pre-registration: `logs/prereg_look_follows_write.md` (thresholds and definitions unchanged). Code: this folder
(written independently of `analysis/investigations/qk/`, which I read only after computing seed 0). All numbers come
from runs of this code. Full numbers: `results.json`. Generated tables: `tables.md` (also pasted below).

## Verdict

**SUPPORTED, by the literal pre-registered reading, but the margin is one head.** Seeds 1-4 have 8 LOCAL heads. P1-P3
all hold for 7 of them (87.5%; the bar is 80%): P1 holds for 7/8, P2 for 8/8, P3 for 8/8. P4 (reported separately)
holds for 7/8.

The verdict is fragile:
- **attn_s1/h3 passes P1 only because the pre-registered ceiling has a negative gain.** Adding the exact token-token
  term makes that head's output worse: TT gain -0.052 +- 0.011 (paired SE over the 32 eval windows), against an own-UOV
  gain of +0.092 +- 0.004. Literally, 0.092 >= 0.5 x (-0.052), so P1 holds. If P1 is counted as failed whenever the
  ceiling's gain is negative, 6/8 = 75% pass and the verdict becomes **MIXED** (P1 fails for 2/8, not REFUTED).
- **attn_s3/h1 fails P1**: own gain 0.064 against TT gain 0.168 (ratio 0.38).
- Two other confirmatory heads have own/TT ratios of 0.76 (s3/h3) and 0.84 (s4/h3), against the threshold 0.5.

The head-specificity parts are clear-cut. Other heads' UOV gains and the token-shuffled gains are at most 0.10 x the own
gain (P2), and the own-UOV pure-pair correlation is 0.34-0.44 on every LOCAL head, against at most 0.08 for any other
head (P3).

| model/head | own gain | TT gain (pre-reg ceiling) | P1 (own/TT) | P2 (max control / own) | P3 (own corr; max other) | P4 c1, c2 (ratio to median) |
|---|---|---|---|---|---|---|
| attn_s1/h3 | +0.0920 | -0.0524 | yes (-1.76; holds only because TT < 0) | yes (-0.01) | yes (0.404; 0.057) | yes, yes (1.03, 1.00) |
| attn_s2/h2 | +0.1150 | +0.0549 | yes (2.09) | yes (0.04) | yes (0.368; 0.015) | yes, yes (0.90, 0.95) |
| attn_s3/h0 | +0.1121 | +0.0568 | yes (1.97) | yes (0.01) | yes (0.335; -0.003) | yes, yes (1.00, 0.88) |
| attn_s3/h1 | +0.0640 | +0.1683 | **NO (0.38)** | yes (0.10) | yes (0.436; 0.082) | yes, yes (1.21, 1.35) |
| attn_s3/h3 | +0.0886 | +0.1170 | yes (0.76) | yes (0.01) | yes (0.378; 0.018) | yes, yes (0.74, 1.23) |
| attn_s4/h0 | +0.1764 | +0.1416 | yes (1.25) | yes (0.02) | yes (0.371; -0.025) | yes, yes (0.89, 0.85) |
| attn_s4/h2 | +0.1018 | +0.0170 | yes (5.98) | yes (0.03) | yes (0.415; 0.034) | yes, yes (0.84, 0.95) |
| attn_s4/h3 | +0.0707 | +0.0837 | yes (0.84) | yes (0.05) | yes (0.436; 0.059) | **NO** (1.65), yes (1.47) |
| attn_s0/h0 (cross-check) | +0.1024 | +0.0652 | yes (1.57) | yes (0.03) | yes (0.389; 0.028) | NO (0.54), yes (0.69) |
| attn_s0/h2 (cross-check) | +0.1479 | +0.1233 | yes (1.20) | yes (0.05) | yes (0.449; -0.016) | yes, yes (1.27, 1.22) |
| attn_s0/h3 (cross-check) | +0.0702 | +0.1038 | yes (0.68) | yes (0.01) | yes (0.391; 0.021) | NO, NO (1.93, 1.85) |

P4 medians over the 11 LOCAL heads of seeds 0-4: c1 = 0.233, c2 = 0.251 logits per logit of usefulness. 8 of 11 heads
(7 of 8 confirmatory) are within a factor 1.5 for both coefficients. The spread comes from seed 0 (h0 c1 0.126, h3
0.450/0.464) and s4/h3 (0.385/0.370). So the law has a roughly shared constant, but some heads deviate by up to 2x.

### Sensitivity (exploratory, not part of the pre-registered verdict)

- **Other ceiling forms** (`extra_arms.py`): TT only at d = 1..7 with two coefficients (round 2's structure), or TT
  at all distances with separate coefficients for d = 1, 2..7 and >= 8. Taking either one, or the best of the three
  forms, as the ceiling gives the same count: 7/8. s3/h1 still fails (0.40 / 0.36). s1/h3's ceiling stays negative in
  every form (-0.038, -0.040). So the ceiling itself is broken for that head, not just the pre-registered form of it.
  Likely cause (not tested): the TT arm adds the token-token term for earlier keys but not for the self position.
  s1/h3 puts 65% of its attention on itself, so a query-token main effect in TT shifts the self-versus-earlier balance
  by token, and the 4-way query-shape self term cannot undo that. I wrote a variant with TT at the self position too
  (`tt_self` in `extra_arms.py`), but it was not run (see Problems).
- **LOCAL by the per-query-renormalised mean distance** instead of the pooled one: the same 8 heads, the same verdict.
- **Near-threshold RECALL heads.** s1/h1 (12.75 tokens), s1/h2 (12.95) and s2/h0 (13.43) are RECALL by the rule, but
  they behave like LOCAL heads. All three satisfy P1-P3 (own/TT 0.55, 3.75, and a negative TT for s2/h0; own
  correlation 0.31, 0.37, 0.36).
- **Paired SEs** (only for attn_s1 and attn_s2, the two models the per-window rerun reached): own-UOV gains
  +-0.003 to 0.008, TT gains +-0.006 to 0.012. The negative TT gains of s1/h3 (-0.052 +- 0.011) and s2/h0
  (-0.113 +- 0.011) are well outside noise.

## Seed-0 cross-check against round 2 (`analysis/investigations/qk/NOTES.md`, read after computing seed 0)

| quantity (attn_s0) | round 2 | this replication |
|---|---|---|
| pure-pair corr, own UOV, d = 1: h0 / h2 | 0.39 / 0.45 | **0.389 / 0.449** |
| pure-pair corr, other head's UOV: h0 (h2's UOV) / h2 (h0's UOV) | -0.11 / -0.12 | -0.108 / -0.119 |
| UOV tables (mine x final-LN scale vs round 2's `UOV_h{0,2,3}_{d}.npy`, pairs N >= 20) | - | corr 1 - 4e-12, max abs diff 0.004 |
| h3 output KL: generic base / + own UOV / + TT | 0.299 / 0.229 / 0.226 | 0.2994 / 0.2292 / 0.1956 |
| h3 own-UOV coefficients (logit units) | 0.391 / 0.403 raw = 0.450 / 0.463 after x 1.1499 | 0.450 / 0.464 |
| h0 output KL: base / + own / + TT | 0.391 / 0.266 / 0.318 (program rule + key shape base) | 0.3812 / 0.2788 / 0.3160 (generic base) |
| h2 output KL: base / + own / + TT | 0.331 / 0.222 / 0.264 (program LOCAL-2 rule base) | 0.4860 / 0.3381 / 0.3627 (generic base) |
| P1 ratio own/TT: h0 / h2 / h3 | 1.71 / 1.63 / 0.96 | 1.57 / 1.20 / 0.68 |
| P2 max control / own | <= 0.13 | 0.03 / 0.05 / 0.01 |

Agreement: the corpus tables, the pure-pair correlations, and every head-3 quantity that uses the same definitions
agree to about 3 decimals. That covers the base, the own-UOV arm and its coefficients.

The disagreements have identified causes:
1. **Base rule.** Round 2 used the program's head-specific rules as the base for heads 0 and 2. The pre-registration
   uses the generic base for every head, so the absolute KLs and the h0/h2 coefficients differ. The generic base is
   much weaker for h2 (0.486 vs 0.331).
2. **TT ceiling.** Round 2's TT table used centred token embeddings x LN gain / 0.25: no position embedding and no
   per-token normalisation. It applied only at d = 1..7, with two coefficients. Mine uses the task's definition:
   per-token LN1(E + pbar), at all distances, one coefficient. On h3 mine is a stronger ceiling (0.196, or 0.186 with
   round 2's d = 1..7 structure, vs 0.226). That stronger ceiling, not the UOV arm, is why h3's P1 ratio drops from
   0.96 to 0.68.
3. **Units.** Round 2's OV was not divided by the final-LN scale. This changes only coefficient values (x 1.1499), not
   gains or correlations.

## Definition choices (where the pre-registration or the task left room)

1. **OV_h(k, v)**: LN1(E_k + pbar) with pbar = mean of wpe[1..255]. LN uses eps 1e-5, gain g1, no bias. Then W_V^h and
   W_O^h, subtract the mean over d_model, multiply by the final-LN gain, unembed with wte, and divide by the typical
   final-LN scale. That scale = mean over all 32 x 256 positions of sqrt(var + eps) of the final-LN input on
   train_windows(32, 2024). For mlp_s0 the MLP is included in that input, but OV is the direct path only.
   **Typical scale:** attn_s0 1.1499, attn_s1 1.2142, attn_s2 1.3175, attn_s3 1.1020, attn_s4 1.2106, mlp_s0 0.8703.
2. **UOV**: S and N are summed over every stream position j = 0 .. L-d-2 of the 527,489,811-token TRAIN stream,
   including pairs that straddle the end-of-story token, as attention does. N is recounted in the same pass on the
   same positions. P1 = exp(stats().log_follow). Shrinkage 20. Chunks are 10M positions; there is one weighted
   bincount per (head, d, chunk).
3. **N_d check**: the recount of N_1 on the first 25M tokens never exceeds the file's count in any cell. Its
   correlation with the whole-stream file is 0.99996, and the median cell ratio to the expected share is 0.997. The
   full-stream recount of every N_d (d = 1..7) differs from `N_d.npy` by at most 1, in at most 7 cells per d: one cell
   from the last-position convention, the rest from float32 rounding of counts above 2^24 in the file. My own counts
   are used.
4. **Token-shuffled control**: OV_h is replaced by OV_h[pi] in both S and B, with pi = default_rng(0).permutation(2000).
   That gives B' = B[pi], and S' is recounted with the permuted vote.
5. **TT_h(q, k)** = (LN1(E_q + pbar) W_Q^h) . (LN1(E_k + pbar) W_K^h) / sqrt(32), applied at every j < i with one
   coefficient (not at j = i).
6. **Base rule**: 9 distance buckets for j < i (1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+), self x query shape (4), key
   shape (4) for j < i. The pre-registration's distance-0 bucket is dropped because it equals the sum of the four
   self x shape indicators (identical function class). Shapes come from `stats().shape`: word start, continuation,
   other, end of story. The fit minimises the mean pattern cross-entropy over queries i >= 1 (convex, unregularised)
   with L-BFGS and a strong-Wolfe line search, float64. All fits converged, with max |gradient| <= 2e-6.
7. **Pair arms**: c1 x UOV_h,1(t_{i-1}, t_i) at d = 1 and c2 x UOV_h,d(t_{i-d}, t_i) at d = 2..7. Each distance uses
   its own table. Fit windows are train_windows(32, 2024).
8. **Output test**: the pattern of the fitted rule on train_windows(32, 2025) replaces that head's pattern in an
   explicit fp32 forward pass. Everything else is exact, including the head's values and W_O. The substituted pass
   reproduces the model to 7e-5 in logits (the explicit-vs-SDPA difference), and the result is identical to the
   explicit pass. KL(model || substituted) is averaged over all 32 x 256 positions, position 0 included.
9. **LOCAL**: pooled attention-weighted mean distance over (i >= 1, j < i) on train_windows(32, 2024), LOCAL if
   <= 10. The per-query-renormalised version gives the same classification at 10.
10. **Pure-pair correlation**: the exact score for key k at position i-1 and query q at position i, minus the query's
    self score at position i, averaged over i in {16, 48, 96, 160, 224}. The numpy formula matches the model's scores
    to 3e-5. Pairs with N_1 >= 20 (278,269 pairs) are weighted by N_1. Double centring is a weighted additive fit
    a_k + b_q, solved by preconditioned CG to a relative residual of 1e-13. The weighted correlation of residuals
    follows. The self-score subtraction is a query main effect, so it has no effect after double centring.
11. **P1-P4 as written**: P1 own gain >= 0.5 x TT gain, literal even when the TT gain is negative. P2 max(other
    heads' gains, shuffled gain) <= 0.25 x own gain. P3 own correlation >= 0.25 and above every other head's UOV
    correlation; the shuffled control is not part of P3. P4 checks that coefficient / median lies in [1/1.5, 1.5],
    separately for c1 and c2, with the median over all 11 LOCAL heads of seeds 0-4. Verdict rule as pre-registered.

## Classification

| model | mean attention distance, heads 0-3 (pooled) | LOCAL heads |
|---|---|---|
| attn_s0 | 4.98, 27.35, 4.35, 9.86 | 0, 2, 3 |
| attn_s1 | 25.42, 12.75, 12.95, 4.08 | 3 |
| attn_s2 | 13.43, 17.31, 6.27, 29.00 | 2 |
| attn_s3 | 3.58, 6.64, 30.21, 6.62 | 0, 1, 3 |
| attn_s4 | 6.49, 30.84, 6.15, 8.36 | 0, 2, 3 |
| mlp_s0 (exploratory) | 8.87, 5.53, 3.25, 36.83 | 0, 1, 2 |

## Exploratory findings

- **RECALL heads (d = 1..7 protocol, no prediction).** Six heads clearly look far (mean distance >= 17): s0/h1, s1/h0,
  s2/h1, s2/h3, s3/h2 and s4/h1. On them, short-range own UOV recovers only a small part of the TT gain (own/TT
  0.06-0.38). It stays head-specific (P2 holds on all 9 RECALL heads), and its pure-pair correlation at d = 1 is
  0.15-0.34. The distance-band version (UOV aggregated over d = 8..63) was not run (see Problems).
- **mlp_s0 (OV through the direct path only).** The law does not carry over in this form. Own-UOV gains are 0.0011,
  0.0016 and 0.0020 for heads 1-3, against TT gains of 0.08-0.13. Head 0 gains 0.080 (TT 0.249). Pure-pair
  correlations are 0.07-0.11, and other heads' UOV is comparable (up to 0.10). The likely reason is that in the MLP
  model the heads' outputs act mostly through the MLP, so the direct-path OV is the wrong "what it writes". This is
  not a test of the law for mlp models.

## Problems and notes

- **Machine contention.** After the confirmatory runs, the laptop came under heavy memory pressure: about 9 GB in the
  memory compressor while other jobs ran. The same steps then ran 20-80x slower (the mlp_s0 corpus pass took 55 min
  instead of 3; one fit_arms model took 65 min instead of 1). I stopped the exploratory queue. Not run: the `tt_self`
  ceiling variant, the per-window rerun for attn_s0/s3/s4 (paired SEs exist only for s1 and s2), and the 8..63
  distance-band UOV for RECALL heads (code is in place: `uov_tables.py --band 8 63 --max-tokens N`,
  `fit_arms.py --band 8 63`). Where the rerun finished, it reproduced the first run's numbers exactly.
- I ran over the 3-hour budget (about 4 h wall time), almost all of it in the slowed exploratory steps.
- The cache `analysis/out/look_write/` (2.6 GB, git-ignored by its own `.gitignore`) keeps the own-UOV tables for all
  heads of attn_s0..s4 at d = 1..7, d = 1 shuffled tables and N_1. I deleted the d >= 2 shuffled tables and the
  mlp_s0 d >= 2 tables after use.
- No files outside `analysis/look_write/` and `analysis/out/look_write/` were changed; git state untouched.


# Generated tables (summarize.py -> tables.md)

## Per head: classification, base KL, output-test gains, fitted coefficients

Gains = KL(base) - KL(arm), nats/token, exact-model substitution on train_windows(32, 2025). `others` lists the other heads' UOV arms in increasing head order (the head itself skipped). c1, c2 = own-UOV coefficients for d = 1 and d = 2..7; c_TT = coefficient of the exact TT term.

| model | head | kind | mean dist | base KL | own | others | perm | TT | c1 | c2 | c_TT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_s0 | 0 | LOCAL | 4.98 | 0.3812 | +0.1024 | -0.0007 / +0.0011 / +0.0028 | +0.0001 | +0.0652 | 0.126 | 0.174 | 0.387 |
| attn_s0 | 1 | RECALL | 27.35 | 0.2485 | +0.0184 | -0.0000 / -0.0000 / +0.0002 | +0.0001 | +0.2071 | 0.172 | 0.266 | 0.825 |
| attn_s0 | 2 | LOCAL | 4.35 | 0.4860 | +0.1479 | +0.0004 / +0.0000 / +0.0079 | +0.0003 | +0.1233 | 0.295 | 0.305 | 0.426 |
| attn_s0 | 3 | LOCAL | 9.86 | 0.2994 | +0.0702 | +0.0004 / +0.0005 / -0.0002 | -0.0006 | +0.1038 | 0.450 | 0.464 | 0.573 |
| attn_s1 | 0 | RECALL | 25.42 | 0.1965 | +0.0290 | +0.0014 / +0.0004 / -0.0001 | +0.0016 | +0.1487 | 0.257 | 0.264 | 0.839 |
| attn_s1 | 1 | RECALL | 12.75 | 0.3750 | +0.0844 | -0.0000 / -0.0005 / +0.0017 | -0.0000 | +0.1539 | 0.418 | 0.468 | 0.525 |
| attn_s1 | 2 | RECALL | 12.95 | 0.3777 | +0.1006 | +0.0031 / -0.0005 / +0.0013 | -0.0006 | +0.0268 | 0.403 | 0.513 | 0.271 |
| attn_s1 | 3 | LOCAL | 4.08 | 0.2470 | +0.0920 | -0.0006 / -0.0018 / -0.0034 | -0.0005 | -0.0524 | 0.239 | 0.251 | 0.258 |
| attn_s2 | 0 | RECALL | 13.43 | 0.4517 | +0.0922 | +0.0003 / -0.0007 / -0.0014 | +0.0022 | -0.1129 | 0.447 | 0.417 | 0.372 |
| attn_s2 | 1 | RECALL | 17.31 | 0.2685 | +0.0418 | -0.0001 / +0.0002 / -0.0004 | +0.0003 | +0.1453 | 0.266 | 0.244 | 0.663 |
| attn_s2 | 2 | LOCAL | 6.27 | 0.4238 | +0.1150 | -0.0015 / +0.0004 / +0.0050 | +0.0000 | +0.0549 | 0.210 | 0.239 | 0.302 |
| attn_s2 | 3 | RECALL | 29.00 | 0.4774 | +0.0643 | +0.0003 / +0.0009 / +0.0004 | +0.0064 | +0.1698 | 0.284 | 0.301 | 0.501 |
| attn_s3 | 0 | LOCAL | 3.58 | 0.3483 | +0.1121 | -0.0010 / -0.0002 / +0.0007 | +0.0002 | +0.0568 | 0.233 | 0.222 | 0.425 |
| attn_s3 | 1 | LOCAL | 6.64 | 0.2396 | +0.0640 | +0.0003 / +0.0064 / +0.0002 | +0.0055 | +0.1683 | 0.281 | 0.338 | 0.744 |
| attn_s3 | 2 | RECALL | 30.21 | 0.2031 | +0.0134 | -0.0002 / +0.0015 / +0.0000 | +0.0000 | +0.1700 | 0.178 | 0.301 | 0.820 |
| attn_s3 | 3 | LOCAL | 6.62 | 0.3863 | +0.0886 | +0.0006 / +0.0008 / +0.0008 | -0.0000 | +0.1170 | 0.173 | 0.309 | 0.483 |
| attn_s4 | 0 | LOCAL | 6.49 | 0.4245 | +0.1764 | +0.0004 / +0.0026 / +0.0025 | +0.0027 | +0.1416 | 0.208 | 0.213 | 0.147 |
| attn_s4 | 1 | RECALL | 30.84 | 0.2206 | +0.0115 | +0.0000 / +0.0002 / +0.0005 | +0.0001 | +0.1866 | 0.174 | 0.312 | 0.811 |
| attn_s4 | 2 | LOCAL | 6.15 | 0.4326 | +0.1018 | -0.0001 / -0.0001 / +0.0028 | +0.0001 | +0.0170 | 0.196 | 0.238 | 0.241 |
| attn_s4 | 3 | LOCAL | 8.36 | 0.2826 | +0.0707 | +0.0023 / +0.0033 / +0.0006 | +0.0006 | +0.0837 | 0.385 | 0.370 | 0.493 |

## Pure-pair correlations at d = 1 (double-centred, weighted by N_1, pairs with N_1 >= 20)

| model | head | kind | own UOV | other heads' UOV | perm | TT |
|---|---|---|---|---|---|---|
| attn_s0 | 0 | LOCAL | +0.389 | +0.028 / -0.108 / -0.046 | +0.007 | +0.999 |
| attn_s0 | 1 | RECALL | +0.176 | +0.033 / +0.027 / -0.010 | -0.014 | +1.000 |
| attn_s0 | 2 | LOCAL | +0.449 | -0.119 / -0.016 / -0.073 | -0.001 | +0.999 |
| attn_s0 | 3 | LOCAL | +0.391 | -0.059 / +0.021 / -0.060 | +0.020 | +0.999 |
| attn_s1 | 0 | RECALL | +0.228 | +0.029 / -0.026 / +0.066 | +0.024 | +1.000 |
| attn_s1 | 1 | RECALL | +0.306 | +0.026 / -0.004 / -0.023 | +0.002 | +0.999 |
| attn_s1 | 2 | RECALL | +0.367 | -0.059 / +0.022 / -0.067 | -0.009 | +0.999 |
| attn_s1 | 3 | LOCAL | +0.404 | +0.057 / -0.017 / -0.074 | +0.003 | +0.999 |
| attn_s2 | 0 | RECALL | +0.362 | -0.030 / -0.056 / +0.072 | -0.017 | +0.999 |
| attn_s2 | 1 | RECALL | +0.341 | -0.011 / +0.029 / -0.049 | -0.020 | +1.000 |
| attn_s2 | 2 | LOCAL | +0.368 | -0.088 / +0.015 / -0.062 | +0.009 | +0.999 |
| attn_s2 | 3 | RECALL | +0.243 | +0.001 / -0.032 / +0.025 | +0.009 | +1.000 |
| attn_s3 | 0 | LOCAL | +0.335 | -0.017 / -0.003 / -0.095 | -0.015 | +0.999 |
| attn_s3 | 1 | LOCAL | +0.436 | -0.045 / +0.082 / -0.058 | -0.021 | +0.999 |
| attn_s3 | 2 | RECALL | +0.172 | +0.065 / +0.007 / -0.008 | +0.005 | +1.000 |
| attn_s3 | 3 | LOCAL | +0.378 | -0.115 / -0.022 / +0.018 | -0.004 | +0.999 |
| attn_s4 | 0 | LOCAL | +0.371 | -0.036 / -0.042 / -0.025 | +0.003 | +0.999 |
| attn_s4 | 1 | RECALL | +0.152 | +0.014 / +0.048 / +0.006 | -0.004 | +1.000 |
| attn_s4 | 2 | LOCAL | +0.415 | -0.027 / +0.034 / -0.117 | -0.016 | +0.999 |
| attn_s4 | 3 | LOCAL | +0.436 | -0.050 / +0.059 / -0.141 | +0.054 | +1.000 |

## Predictions per LOCAL head

| model/head | P1 (own/TT gain) | P2 (max control/own) | P3 (own corr; max other) | P4 c1 (ratio to median) | P4 c2 (ratio) |
|---|---|---|---|---|---|
| attn_s0/h0 (cross-check) | yes (1.57) | yes (0.03) | yes (0.389; 0.028) | NO (0.54) | yes (0.69) |
| attn_s0/h2 (cross-check) | yes (1.20) | yes (0.05) | yes (0.449; -0.016) | yes (1.27) | yes (1.22) |
| attn_s0/h3 (cross-check) | yes (0.68) | yes (0.01) | yes (0.391; 0.021) | NO (1.93) | NO (1.85) |
| attn_s1/h3 | yes (-1.76) | yes (-0.01) | yes (0.404; 0.057) | yes (1.03) | yes (1.00) |
| attn_s2/h2 | yes (2.09) | yes (0.04) | yes (0.368; 0.015) | yes (0.90) | yes (0.95) |
| attn_s3/h0 | yes (1.97) | yes (0.01) | yes (0.335; -0.003) | yes (1.00) | yes (0.88) |
| attn_s3/h1 | NO (0.38) | yes (0.10) | yes (0.436; 0.082) | yes (1.21) | yes (1.35) |
| attn_s3/h3 | yes (0.76) | yes (0.01) | yes (0.378; 0.018) | yes (0.74) | yes (1.23) |
| attn_s4/h0 | yes (1.25) | yes (0.02) | yes (0.371; -0.025) | yes (0.89) | yes (0.85) |
| attn_s4/h2 | yes (5.98) | yes (0.03) | yes (0.415; 0.034) | yes (0.84) | yes (0.95) |
| attn_s4/h3 | yes (0.84) | yes (0.05) | yes (0.436; 0.059) | NO (1.65) | yes (1.47) |

Medians over LOCAL heads of seeds 0-4: c1 = 0.233, c2 = 0.251.

Confirmatory LOCAL heads (seeds 1-4): 8; P1 holds 7, P2 8, P3 8, all of P1-P3 7, P4 7. **Verdict: SUPPORTED.**


## Exploratory: mlp_s0 (OV through the direct path only)

| model | head | kind | mean dist | base KL | own | others | perm | TT | c1 | c2 | c_TT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | 0 | LOCAL | 8.87 | 0.2665 | +0.0805 | -0.0025 / -0.0017 / +0.0001 | -0.0029 | +0.2491 | 0.182 | 0.283 | 0.780 |
| mlp_s0 | 1 | LOCAL | 5.53 | 0.1166 | +0.0011 | -0.0002 / +0.0003 / +0.0001 | -0.0001 | +0.0843 | 0.233 | 0.251 | 0.802 |
| mlp_s0 | 2 | LOCAL | 3.25 | 0.1623 | +0.0016 | +0.0000 / -0.0003 / +0.0005 | -0.0001 | +0.1318 | 0.287 | 0.327 | 0.748 |
| mlp_s0 | 3 | RECALL | 36.83 | 0.1481 | +0.0020 | -0.0000 / +0.0000 / +0.0001 | +0.0000 | +0.1271 | 0.182 | 0.430 | 0.782 |

| model | head | kind | own UOV | other heads' UOV | perm | TT |
|---|---|---|---|---|---|---|
| mlp_s0 | 0 | LOCAL | +0.070 | +0.025 / +0.026 / +0.104 | -0.043 | +1.000 |
| mlp_s0 | 1 | LOCAL | +0.108 | +0.055 / +0.019 / +0.088 | +0.018 | +0.991 |
| mlp_s0 | 2 | LOCAL | +0.078 | +0.019 / +0.026 / +0.071 | -0.005 | +0.992 |
| mlp_s0 | 3 | RECALL | +0.082 | -0.005 / +0.014 / +0.014 | +0.002 | +1.000 |
