# Addendum results (logs/prereg_look_follows_write_addendum.md), one-layer models

## A. Ceiling with the self position (`tt_self`)

- **A1 holds.** With the token-token term also applied at the self position, the ceiling gain is positive for
  attn_s1/h3 (+0.220) and attn_s2/h0 (+0.402); both were negative under the pre-registered ceiling.
- **A2 fails: 0 of 8.** Own-vote usefulness reaches 0.26-0.46 of the `tt_self` gain in the 8 confirmatory LOCAL heads
  (threshold 0.5).

The `tt_self` ceiling is 2-3 times larger than the pre-registered one (0.20-0.45 nats per head) because, per query
token, it sets each earlier key's score relative to the current token's own score. An exploratory split (not
pre-registered; analysis/look_write/extra_arms_self.py) shows:
- the self-position term on its own adds little (-0.04 to +0.09);
- usefulness plus the self term (`uov_self`) gains only slightly more than usefulness alone.

So the gap to the ceiling is the joint, token-specific weighting of earlier keys relative to the current token, and
own-vote usefulness accounts for about a third of it.

**Revised reading of P1:** in the strictest comparison, own-vote usefulness explains about a third (26-46%) of what
the token-token attention term buys in output. Head-specificity (P2, P3) is unaffected.

| head | class | own UOV gain | TT (pre-reg) | tt_self | self only | UOV + self | own / tt_self |
|---|---|---|---|---|---|---|---|
| attn_s1/h0 | RECALL | +0.029 | +0.149 | +0.151 | +0.001 | +0.030 | 0.19 |
| attn_s1/h1 | RECALL | +0.084 | +0.154 | +0.305 | +0.009 | +0.091 | 0.28 |
| attn_s1/h2 | RECALL | +0.101 | +0.027 | +0.311 | -0.003 | +0.137 | 0.32 |
| attn_s1/h3 | LOCAL | +0.092 | -0.052 | +0.220 | +0.015 | +0.102 | 0.42 |
| attn_s2/h0 | RECALL | +0.092 | -0.113 | +0.402 | +0.076 | +0.184 | 0.23 |
| attn_s2/h1 | RECALL | +0.042 | +0.145 | +0.212 | +0.053 | +0.088 | 0.20 |
| attn_s2/h2 | LOCAL | +0.115 | +0.055 | +0.381 | -0.002 | +0.121 | 0.30 |
| attn_s2/h3 | RECALL | +0.064 | +0.170 | +0.329 | +0.062 | +0.127 | 0.20 |
| attn_s3/h0 | LOCAL | +0.112 | +0.057 | +0.324 | +0.070 | +0.176 | 0.35 |
| attn_s3/h1 | LOCAL | +0.064 | +0.168 | +0.200 | +0.015 | +0.077 | 0.32 |
| attn_s3/h2 | RECALL | +0.013 | +0.170 | +0.172 | +0.006 | +0.019 | 0.08 |
| attn_s3/h3 | LOCAL | +0.089 | +0.117 | +0.346 | +0.057 | +0.148 | 0.26 |
| attn_s4/h0 | LOCAL | +0.176 | +0.142 | +0.383 | -0.036 | +0.150 | 0.46 |
| attn_s4/h1 | RECALL | +0.012 | +0.187 | +0.190 | +0.003 | +0.015 | 0.06 |
| attn_s4/h2 | LOCAL | +0.102 | +0.017 | +0.385 | +0.032 | +0.138 | 0.26 |
| attn_s4/h3 | LOCAL | +0.071 | +0.084 | +0.250 | +0.044 | +0.112 | 0.28 |
| attn_s0/h0 | LOCAL | +0.102 | +0.065 | +0.333 | +0.025 | +0.125 | 0.31 |
| attn_s0/h1 | RECALL | +0.018 | +0.207 | +0.215 | +0.017 | +0.035 | 0.09 |
| attn_s0/h2 | LOCAL | +0.148 | +0.123 | +0.454 | +0.087 | +0.223 | 0.33 |
| attn_s0/h3 | LOCAL | +0.070 | +0.104 | +0.252 | +0.068 | +0.135 | 0.28 |

(Output-test gains in nats, exact-model substitution, train_windows(32, 2025).)

## B. Paired standard errors

Re-running fit_arms.py on attn_s0, s3 and s4 reproduces every committed gain exactly (max difference 0.0) and adds the
per-window KLs, so analysis/look_write/results.json now has paired SEs for all 20 heads. Own-usefulness gains have SEs
of 0.002-0.008 nats. The verdict is unchanged (7/8 literal).

## C. Far-looking heads: usefulness over distances 8..63

The addendum said "the six RECALL heads of attn_s1..s4", but those seeds have 8 RECALL heads; the number was a
miscount when writing. Both readings are reported.
- **C1 (own band gain > own d = 1..7 gain): 4 of 8.** Holds literally ("at least 4"); fails the proportional reading
  (4 of 6 = 67%; 4 of 8 = 50%).
- **C2 (controls <= 0.25 x own band gain): 7 of 8** (attn_s3/h2: 0.019 vs 0.067, ratio 0.28).

The band version wins exactly for the truly far-looking heads (mean distance 25-31 tokens: s1/h0, s2/h3, s3/h2, s4/h1,
and s0/h1) and loses for the four near-threshold heads (12-17 tokens: s1/h1, s1/h2, s2/h0, s2/h1), which behave like
local heads (they also met P1-P3 in the replication). So the usefulness of a head's own vote also tracks where
far-looking heads look, when it is measured over the distances they actually use. The gains are smaller (0.05-0.07
nats) than the full token-token term's (0.15-0.21).

| head | own d 1..7 | own band 8..63 | max control | TT | C1 | C2 |
|---|---|---|---|---|---|---|
| attn_s1/h0 | +0.029 | +0.047 | +0.010 | +0.149 | yes | yes |
| attn_s1/h1 | +0.084 | +0.011 | +0.001 | +0.154 | no | yes |
| attn_s1/h2 | +0.101 | +0.020 | +0.000 | +0.027 | no | yes |
| attn_s2/h0 | +0.092 | +0.009 | -0.002 | -0.113 | no | yes |
| attn_s2/h1 | +0.042 | +0.020 | +0.002 | +0.145 | no | yes |
| attn_s2/h3 | +0.064 | +0.073 | +0.011 | +0.170 | yes | yes |
| attn_s3/h2 | +0.013 | +0.067 | +0.019 | +0.170 | yes | no |
| attn_s4/h1 | +0.012 | +0.067 | +0.016 | +0.187 | yes | yes |
| attn_s0/h1 (dev) | +0.018 | +0.070 | +0.018 | +0.207 | yes | no |

## D. Echo test: does the relation come before calibration?

attn_s1..s4, the 8 confirmatory heads, checkpoints 229, 718, 2246, 7032, 22009 and final (5M-token sample passes on
MPS; analysis/look_write_deep/echo_traj.py; analysis/out/look_write_deep/echo_traj.json). r(c) = pure-pair correlation
of the head's attention with its own usefulness at that checkpoint; m(c) = correlation of usefulness with its
model-echo version (calibration index).

| checkpoint | LR / peak | dev loss (final 2.534) | r median (min) | m median |
|---|---|---|---|---|
| 229 | 0.78 | 3.556* | 0.049 (0.008) | 0.04 |
| 718 | 1.00 | 3.556* | 0.230 (0.185) | 0.69 |
| 2246 | 0.99 | 2.882 | 0.413 (0.367) | 0.91 |
| 7032 | 0.90 | 2.669 | 0.458 (0.435) | 0.95 |
| 22009 | 0.31 | 2.566 | 0.447 (0.385) | 0.97 |
| final | 0.10 | 2.534 | 0.434 (0.361) | 0.97 |

(* nearest dev evaluation, step 500.)

- **D1 holds.** At step 2246 (LR 99% of peak, dev loss 0.35 above final) all 8 heads have r >= 0.2 (0.37-0.52); at step
  718, 6 of 8.
- **D2 fails.** Pooled over heads and transitions, corr(attention change c -> c+1, usefulness at c) = +0.002 against
  corr(usefulness change, attention at c) = +0.090.
- **D3 holds.** r at step 2246 (median 0.41) is 95% of r at the end (0.43).

**Verdict (as pre-registered): "mutual consistency", not "comes from".** By transition: while the relation forms
(229 -> 718 -> 2246), both directions are small and about equal (+0.06/+0.06; +0.03/+0.04; forward ahead in 4 and 6 of 8
heads). Afterwards the reverse direction dominates and grows (+0.04, +0.09, +0.23; forward ahead in 1, 0, 0 of 8): late
training tunes the vote to where the head already looks, the calibration echo. The calibration index rises together
with r (m = 0.69 already at step 718), so these checkpoints cannot separate the two early on. Attention and vote form
together; what is established is that attention is predictable from the vote and the data, at every stage after step
229.

## Addendum 2: growth test (logs/prereg_look_follows_write_addendum2.md)

attn_s1..s4, the 8 confirmatory heads; analysis/look_write_deep/growth.py; analysis/out/look_write_deep/growth*.json.

| head | G1 at 0.8: vote / attention settle | G2 (pre-registered) | G2, current attention matched (10 / 50 bins) | G3 (pre-reg) | G3, matched (10 / 50) |
|---|---|---|---|---|---|
| s1/h3 | 4807 / 10286 yes | +0.400 | +0.222 / +0.208 | +0.083 | +0.009 / +0.021 |
| s2/h2 | 7032 / 10286 yes | +0.423 | +0.247 / +0.240 | +0.060 | -0.049 / -0.074 |
| s3/h0 | 7032 / 10286 yes | +0.408 | +0.207 / +0.194 | +0.088 | +0.011 / -0.005 |
| s3/h1 | 10286 / 10286 no | +0.360 | +0.175 / +0.167 | +0.137 | +0.019 / +0.028 |
| s3/h3 | 4807 / 10286 yes | +0.359 | +0.178 / +0.171 | +0.119 | +0.001 / -0.006 |
| s4/h0 | 7032 / 10286 yes | +0.503 | +0.275 / +0.259 | +0.085 | -0.008 / -0.013 |
| s4/h2 | 7032 / 7032 no | +0.541 | +0.270 / +0.253 | +0.114 | +0.017 / +0.034 |
| s4/h3 | 15046 / 10286 no | +0.347 | +0.206 / +0.193 | +0.161 | +0.002 / -0.019 |

(G2 and G3 values: mean usefulness or vote similarity of entering pairs minus frequency-matched controls; every G2
value has p = 0.0005, the smallest possible with 2000 permutations.)

- **G1 fails narrowly: 5 of 8** (predicted >= 6). At threshold 0.5 also 5 of 8; at 0.9, 2 of 8. Votes settle at steps
  4807-15046 and attention at 7032-10286: no clear "write before look".
- **G2 holds: 8 of 8.** Pairs that newly become strongly attended had higher usefulness at the earlier checkpoint.
  Exploratory control (not pre-registered; added because the pre-registered design matched only on frequency, and
  pairs entering the top 10% were usually just below it, where usefulness is high simply because usefulness and
  attention already correlate): matching also on the current attention level, in 10 or 50 quantile bins, halves the
  effect but keeps it in all 8 heads (+0.17 to +0.28, p = 0.0005). Usefulness predicts which pairs gain strong
  attention beyond their current attention.
- **G3 (exploratory) holds 8 of 8 as pre-registered, but not under the same control (2 of 8 with 10 or 50 bins).** The
  "entering keys vote like the core keys" effect was mostly the attention confound; recursive generalisation of the
  key set is not supported.

**Direction, combining D and G.** Attention and vote form together (G1 no clear order; D: the relation is near full
strength by step 2246). Which pairs a head comes to attend strongly is predicted by the usefulness of its vote there,
beyond the current attention (G2 with controls), while across the bulk of weakly attended pairs, and late in
training, the vote adapts to where the head looks (D2 reverse direction). The best description is co-adaptation:
"where a head looks" and "what it writes" shape each other, with usefulness leading at the top of the attention
distribution. A strict "comes from" (the vote first, attention following) is not supported.
