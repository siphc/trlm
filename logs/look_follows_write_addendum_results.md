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

## C, D

Pending (C: distance band 8..63; D: echo test on checkpoints).
