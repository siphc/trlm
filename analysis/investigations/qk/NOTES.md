# qk: pair-specific attention in heads 0 and 2 (and 3), 2026-10-13

Question: the attention rules of LOCAL-0 / LOCAL-2 depend on features of the current token and of the earlier token,
not on the pair. Is the missing pair term "the earlier token is informative about the next token given the current
token" (conditional MI / trigram gain), and which corpus statistic best predicts the exact QK pair interaction?

All numbers below come from runs in this folder. Windows: attention rules fitted on train_windows(32, 2024) (the
program's fit windows); pattern KL reported on held-out train_windows(16, 2025); exact-model substitution on
train_windows(32, 2025); program output with analysis/vote_fit.py (fit 32 / eval 64 TRAIN windows, seed 2000, head R^2
on 16 windows) unless noted.

## Short answer

1. **The heads' pair-specific attention follows the usefulness of their own vote.** Head h reads earlier token k
   (d tokens back) more when the head's own OV vote for k fits what actually follows (k ... q) better than what the
   current token q alone predicts:
   `UOV_h,d(k, q) = E[OV_h(k, next) | k sits d before q] - E_{P1(.|q)}[OV_h(k, .)]`, counted over the TRAIN corpus.
   Adding this as one attention feature (2 scalars: d = 1 and d = 2..7) and substituting the rule's pattern into the exact
   model removes as much output KL as the exact token-token QK term, or more: head 0 0.391 -> 0.266 (exact TT 0.318),
   head 2 0.331 -> 0.222 (TT 0.264), head 3 0.299 -> 0.229 (TT 0.226). This holds only for the head's own vote:
   the other heads' OV usefulness gives -0.002 (h0), -0.014 (h2) and +0.000 (h3).
2. **Corpus statistics only approximate it.** The best corpus pair statistic is the same quantity with the program's
   skip-gram vote (U below). Conditional MI or trigram information gain, PMI, backward probability, bigram votes, copy
   vetoes and transformed PMI votes are all weaker. U recovers about 44% (head 0) and 18% (head 2) of what own-vote
   usefulness recovers in the exact model.
3. **For the program's output, attention is not the bottleneck.** Even the model's exact head-0 and head-2 patterns,
   read through the program's skip-gram votes, gain only 0.0063 +- 0.0013 nats (head 2 alone 0.0068, head 0 alone
   0.0004). The best corpus candidate (head 2 + U) gains 0.0045 +- 0.0003, about two thirds of that ceiling but below
   the 0.005 bar. A second eval set gives 0.0039 +- 0.0003. **No candidate adopted.**
4. **Explain-vs-fit split.** Faithful attention raises how much of the heads' earlier-token effect the program explains,
   without improving output KL. Exact attention raises head 2's R^2 from 0.124 to 0.209 and head 0's from 0.065 to
   0.099. Own-vote-usefulness attention (weights) gets head 0 to 0.098 and head 2 to 0.170. Corpus U barely moves R^2
   (head 2 0.131 vs 0.129 for a 3-direction control). The skip-gram votes are what limit both fidelity and explanation.
   Better votes would make pair-aware attention pay off, and the usefulness rule then gives that attention for free.

## Tools and tables (rebuild order)

- `pairstats.py`: one pass over the 527M-token TRAIN stream (20M-token chunks, 68 s). For d = 1..8: `N_d[k, q]`
  (k exactly d tokens before q, whole stream) and `S_d[k, q]` (sum of PMI_{min(d+1,8)}(k, next token after q)).
  With `--ov` it also writes weights-derived OV tables (insight only).
- `build_tables.py`: `U_d`, `A_d`, `G_d`, `BACK_d` (d = 1..7) and `IG_1` (trigram gain from Stats.tri_*), a few seconds.
- `pairstats_ov.py` (insight, weights): `UOV_h_d` for heads 0, 2 and 3, d = 1..7 (93 s).
- `pairstats2.py`: `UBIG_d` (bigram-vote usefulness), `UCOPY_d` (copy-veto usefulness). `pairstats3.py`: transformed
  votes (UPOS/UCLIP/UPROB; tables deleted after the dead end, rebuild in about 1 min).
- `qk_pairs.py`: the reverse question (correlation with the exact QK pair interaction).
- `qklib.py` + `attnfit.py`: log-linear attention rules (convex pattern cross-entropy fit) for heads 0, 2 and 3 with
  optional pair features. Results are in `attnfit_h{0,2,3}.json`.
- `subst.py`: rule pattern substituted into the exact model, output KL (`subst.json`, `subst_cross.json`,
  `subst_condinfo.json`).
- `make_candidates.py`: writes `analysis/candidates/qk_*.py`. vote_fit logs are in `vf_*.log`, JSON results in
  `analysis/out/candidates/<name>.json`.
- Cache: `analysis/out/candidates/qk/` (1.3 GB; N_d and S_d are the sources; everything else is derived).

Counting rules (all closed-form from the TRAIN stream, using the program's own P1 = exp(log_follow) and PMI tables):

```
N_d(k, q) = #{j : x_j = k, x_{j+d} = q}                                (whole stream, d = 1..8)
S_d(k, q) = sum over those j of PMI_{d+1}(k, x_{j+d+1})
B_d(k, q) = sum_v P1(v | q) PMI_{d+1}(k, v)                            (k's vote, scored by what q alone predicts)
U_d(k, q) = (S_d - N_d B_d) / (N_d + 20)                               "vote usefulness given the current token"
A_d(k, q) = (S_d + 20 B_d) / (N_d + 20)                                "vote fit" (pair-refined info_d(k))
G_d(k, q) = A_d - log sum_v P1(v | q) exp(PMI_{d+1}(k, v))             expected log-lik gain of adding k's vote
IG_1(k, q) = KL(P3(. | k, q) || P1(. | q)), P3 Dirichlet(20) towards P1 (trigram information gain = CMI term)
BACK_d(k, q) = log((N_d + 10 P(k)) / (sum_k' N_d(k', q) + 10))
```

## 1. Reverse question: which statistic predicts the exact QK pair interaction? (qk_pairs.json)

The target is the exact score of key k at distance d for query q, averaged over query positions 16/48/96/160/224,
minus the query's self score. Pairs with count >= 20 (278k at d = 1, 663k at d = 2) are weighted by count. The
double-centred view removes query and key main effects with a weighted additive fit, leaving the pure pair
interaction. At d = 1 the pure pair part is 21% (h0) and 16% (h2) of the raw variance, and the key effect is a
further 19% (h0) and 20% (h2).

Weighted correlation with the double-centred (pure pair) interaction:

| statistic | h0 d=1 | h2 d=1 | h0 d=2 | h2 d=2 |
|---|---|---|---|---|
| PMI_d(k, q) (= log P(q\|k), log P(k\|q) after centring) | 0.18 | 0.15 | 0.17 | 0.12 |
| trigram information gain IG_1 (CMI term) | 0.06 | 0.07 | - | - |
| vote fit A | 0.14 | **0.25** | 0.14 | 0.19 |
| vote usefulness U | 0.14 | 0.19 | 0.18 | **0.20** |
| expected log-lik gain G | 0.11 | 0.17 | 0.13 | 0.16 |
| current is piece x key cont | 0.09 | 0.10 | **0.19** | 0.14 |
| [weights] own OV usefulness | **0.39** | **0.45** | 0.24 | 0.27 |
| [weights] own OV fit | **0.42** | **0.45** | **0.31** | **0.32** |
| [weights] other head's OV usefulness | -0.11 | -0.12 | 0.01 | -0.06 |

(Row-centred view, key effects kept, h0 d=1: U 0.26, BACK 0.25, PMI 0.23, IG 0.12, own OV fit 0.36.) Each head's pair
interaction lines up with its own OV vote, not with another head's. Among corpus statistics, the vote-based ones (A,
U) beat PMI and the trigram gain. The card's numbers (PMI 0.38, IG 0.19) used a different target (interaction centred
over keys only, not over queries).

## 2. Attention-pattern fits (pattern KL on held-out windows; lower is better)

| rule | head 0 | head 2 | head 3 |
|---|---|---|---|
| program rule (h0 8 / h2 12 scalars; h3: generic 19-scalar distance+shape rule) | 0.417 | 0.452 | 0.917 |
| + key shape (h0 only, +3) | 0.384 | - | (in base) |
| + U (2 scalars) | 0.350 (with key) / 0.390 (U*, 1 scalar, no key) | 0.440 (U*: 0.440) | 0.889 |
| + A | 0.356 | 0.446 | |
| + G | 0.366 | 0.441 | |
| + PMI | 0.360 | 0.450 | |
| + BACK | 0.374 | 0.448 | |
| + IG_1 (trigram gain, d=1) | 0.369 | 0.446 | |
| + U + IG_1 | | 0.440 | |
| + all six corpus pair features (12-13 extra) | 0.337 | 0.433 | |
| + UBIG (bigram vote) | 0.374 | 0.449 | 0.915 |
| + UCOPY (copy veto) | 0.383 | 0.450 | 0.902 |
| + U + UBIG + UCOPY | 0.348 | 0.436 | 0.870 |
| + UPOS / UCLIP / UPROB (transformed votes) | 0.359 / 0.355 / 0.371 | 0.441 / 0.440 / 0.452 | 0.892 / 0.890 / 0.912 |
| KEY_INFO replaced by U* (same 12 scalars) | - | 0.443 | - |
| [weights] + own OV usefulness | 0.349 | 0.402 | 0.850 |
| [weights] + another head's OV usefulness | 0.383 / 0.384 | 0.451 / 0.450 | 0.915 / 0.917 |
| [weights] + exact token-token QK (ceiling) | 0.287 | 0.353 | 0.637 |

The fitted U coefficients are stable across heads and distances, about 0.3 per nat (h0 0.36/0.31, h2 0.30/0.32,
h3 0.32/0.33).

## 3. Exact-model pattern substitution (output KL vs the model, 32 TRAIN windows, head's own OV kept)

| rule | head 0 | head 2 | head 3 |
|---|---|---|---|
| program rule | 0.379 | 0.331 | 0.299 (generic base) |
| + key shape | 0.391 | - | - |
| + U* (1 scalar) | 0.361 (-0.018 +- 0.005) | 0.311 (-0.020 +- 0.004) | |
| KEY_INFO -> U* | - | 0.314 (-0.017 +- 0.005) | |
| key + U (head 3: base + U) | 0.336 (-0.055 +- 0.007 vs key) | | 0.283 (-0.016 +- 0.003) |
| [weights] + own OV usefulness | **0.266** (-0.126 +- 0.007) | **0.222** (-0.109 +- 0.005) | **0.229** (-0.070 +- 0.005) |
| [weights] + another head's OV usefulness | 0.389 (-0.002 +- 0.001) | 0.317 (-0.014 +- 0.002) | 0.299 (+0.000) |
| [weights] + exact token-token QK | 0.318 | 0.264 | 0.226 |

Own-vote usefulness has a worse pattern KL than the exact TT term but a better output KL. It gets attention right on
the keys whose votes matter, as one would expect if the QK circuit learned "attend where my value helps".

## 4. Program output (vote_fit; every vote strength refitted; paired on held-out TRAIN windows)

Head R^2 = pooled weighted R^2 of the model head's earlier-token effect. The control rows re-express the program's
own rule as a candidate with 3 directions (self/near/far), which raises R^2 slightly by itself.

| candidate | heads | paired diff +- SE | R^2 program -> +cand | verdict |
|---|---|---|---|---|
| qk_h0_prog (control) | 0 | +0.0000 +- 0.0000 | 0.0645 -> 0.0670 | control |
| qk_h0_key (key shape, no pair) | 0 | +0.0028 +- 0.0003 | 0.0645 -> 0.0701 | reject (worse) |
| qk_h0_key_use (key + U) | 0 | +0.0003 +- 0.0005 | 0.0645 -> 0.0647 | reject |
| qk_h2_prog (control) | 2 | +0.0000 +- 0.0000 | 0.1238 -> 0.1294 | control |
| qk_h2_use (U, 2 scalars) | 2 | -0.0046 +- 0.0003 | 0.1238 -> 0.1313 | reject (below 0.005) |
| qk_h2_use1 (U*, 1 scalar) | 2 | -0.0045 +- 0.0003; 128 win seed 3001: -0.0039 +- 0.0003 | 0.1238 -> 0.1312 (32 win: 0.1232 -> 0.1292) | reject (below bar) |
| qk_h2_condinfo (KEY_INFO -> U*, 12 scalars) | 2 | -0.0045 +- 0.0004; 128 win: -0.0038 +- 0.0004 | 0.1238 -> 0.1290 | reject (below bar) |
| qk_h02_use (h0 key+U and h2 U) | 0, 2 | -0.0010 +- 0.0006 | 0.2144 -> 0.2172 | reject |
| DIAG exact head-0 + head-2 patterns | 0, 2 | -0.0063 +- 0.0013 | 0.2144 -> 0.2472 | ceiling (weights) |
| DIAG exact head-2 pattern | 2 | -0.0068 +- 0.0008 | 0.1238 -> 0.2090 | ceiling (weights) |
| DIAG exact head-0 pattern | 0 | -0.0004 +- 0.0007 | 0.0645 -> 0.0986 | ceiling (weights) |
| DIAG h2 + own OV usefulness | 2 | -0.0037 +- 0.0004 | 0.1238 -> 0.1700 | diagnostic (weights) |
| DIAG h0 key + own OV usefulness | 0 | +0.0014 +- 0.0005 | 0.0645 -> 0.0979 | diagnostic (weights) |

## Hypotheses and verdicts

- **H1. Conditional MI / trigram information gain IG_1(k, q) drives pair attention.** Dead end. Its pure-pair
  correlation is 0.06-0.07. As an attention feature it reaches head 0 0.369 (key 0.384, U 0.350) and head 2 0.446
  (base 0.452, U 0.440), and adds nothing on top of U (h2 0.4399 -> 0.4396). The CMI form measures how much k changes
  the next-token distribution. The heads care about whether their own vote from k points the right way.
- **H2. Vote usefulness U_d(k, q) (expected skip-gram gain given the current token) drives pair attention.** It is
  the best corpus statistic for all three local heads (patterns; exact-model output -0.016 to -0.055). In the program it
  gains 0.0045 for head 2 (below the bar) and nothing for head 0, and raises R^2 by at most about 0.002 over the
  control. Insight-only; candidate rejected.
- **H3. Heads attend where their OWN vote is useful (weights, insight).** Confirmed for heads 0, 2 and 3: correlation
  0.39-0.45 with the pure pair interaction at d = 1, output substitution better than the exact TT term, and head-specific
  (cross-head controls null).
- **H4. Replace head 2's unconditional key informativeness info_d(k) by the conditional U_d(k, q).** Same scalar
  count, pattern 0.452 -> 0.443, output -0.0045 +- 0.0004 (128 windows: -0.0038). A cleaner rule with the same
  scalar count, but below the bar. Inconclusive / insight.
- **H5. Bigram-vote usefulness or copy-veto usefulness explains the pair term** (head 0's OV has a strong anti-copy).
  Dead end: UBIG and UCOPY are weaker than U (h0 0.374 / 0.383, h2 0.449 / 0.450) and add at most 0.002-0.005 to it.
- **H6. Transformed skip-gram votes** (positive PMI, PMI floored at -1, probability votes) predict attention better.
  Dead end: none beats plain log-PMI U on any head.
- **H7. Key-shape salience for head 0** (a non-pair control). Pattern 0.417 -> 0.384, but the program's output is
  worse (+0.0028 +- 0.0003) and the exact-model output is worse too (0.379 -> 0.391). In the program the extra
  attention on word pieces is compensated elsewhere.
- **H8. Better attention closes the heads-0/2 gap in the program.** Refuted by the ceiling: exact attention gains at
  most 0.0068 nats of output KL under skip-gram votes. The gap is in the votes.

## What this says about how the model works

- Beyond their position and shape rules, heads 0, 2 and 3 choose which recent token to read by a single principle:
  read k when the head's own vote for k fits what comes next better than the current token's own prediction. The QK
  pair term is the expected value of the OV vote given the query, as one would expect from training: a key's score
  rises where its value vector points along the prediction error.
- So the pair term carries little information of its own: once you know what a head writes from each token (its
  OV), where it looks follows from corpus counts. The corpus rule is only as good as the program's model of the vote.
  The skip-gram PMI vote is a poor model of what heads 0 and 2 write (cards: OV vs corpus R^2 <= 0.24). That is why
  corpus statistics recover only about 20-45% of the pair effect, and why exact attention barely helps the program's output.
- Head 2's existing rule is already a usefulness rule with the unigram as baseline: SELF_INFO * info_0(q) is the
  usefulness of the self vote and KEY_INFO * info_d(k) that of an earlier token's vote. Conditioning on the current
  token (U) is the same idea, refined.
- Head 3 follows the same principle (own-OV usefulness 0.299 -> 0.229, equal to its exact TT term). Its attention is
  therefore explainable once its vote is known. Its vote is not a skip-gram, which matches the earlier dead end of a
  head-3 skip-gram reader.

## Recommendations for the lead

- Nothing to adopt now. The best corpus candidate (qk_h2_use1 / qk_h2_condinfo) is a consistent 0.004-0.0045 gain,
  15 SE but below 0.005.
- If a better corpus vote rule for heads 0 and 2 is found (e.g. conditional skip-grams), recompute U with that vote,
  `U = E[vote_k(next) | k, q] - E_{P1(.|q)}[vote_k]`. It is one bincount pass per distance plus one dense matmul. Expect
  the attention side to improve with it: with the true vote, the same 1-2 scalar rule gives an output substitution as
  good as the exact QK term. This is also a test for any proposed vote rule: a vote rule that is right should predict
  where the head looks.
- Compute cost if adopted: 7 tables of 2000 x 2000 float32 (112 MB); one 70 s corpus pass plus seconds of matmuls.
