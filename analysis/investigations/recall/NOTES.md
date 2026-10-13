# What is missing from RECALL (program/head1.py)? Investigation "recall", 2026-10-13

## Assumptions

- Model attn_s0, program at tag mechanism-v1. All numbers are on TRAIN windows (`analysis.fitlib.train_windows`).
  The diagnostics use 24 windows with seed 3000 (D1, D2, D7, D10) or 16 windows with seed 2000 (D4). The attention
  fits use 32 windows with seed 2024 and are scored on 16 held-out windows with seed 2000. vote_fit used its
  defaults (eval 64, seed 2000), except where `--fit` is given.
- "R^2" is always the pooled, probability-weighted R^2 of model head 1's earlier-token effect
  (y = lp - lp_ablated under harness 'earlier' ablation), as in analysis/head_effects.py, with directions centred
  over positions. vote_fit's R^2 uses only 16 or 24 windows, so it is noisy at the ±0.01 level.
- Exact model internals (head-1 pattern, OV logit votes, the token-token score term TT) were used **for insight
  only**. lib.py loads them through analysis/scratch/head1_card/h1lib.py, and OV.npy and TT.npy are the head-1
  card's caches. No candidate meant for the program uses them. The ceiling candidate `recall_attn_TT_exact` is
  labelled insight-only.
- The band-25..64 table in `recall_votes_b25` / `fnb` is the head-1 card's cached corpus table
  (analysis/scratch/head1_card/PMI_25_64.npy: same counting rule as the topic table, other distance band). It
  was used for screening only. Nothing passed, so no prepare() counting was written.

## Scripts (analysis/investigations/recall/)

| script | what it measures |
|---|---|
| d1_decompose.py | 2x2 split: {exact, rule} attention x {exact OV, program votes}; by position class; by attended group |
| d2_votes.py | vote content under EXACT attention: bands, 64-class strengths; how well each group's votes are described |
| d3_ovrows.py | OV rows vs corpus tables (unweighted row R^2; weak screen) |
| fit_attn.py | fits attention-rule variants to head 1's own pattern (pattern cross-entropy); held-out pattern KL |
| d4_pattern.py | where the rule's pattern differs: by query class, key group, distance, entropy, under/over-attended tokens |
| d5_examples.py | example rows: head 1's top keys vs the rule's top keys |
| d6_loudness.py | whether over-attended keys are quiet (sinks) |
| d7_attn_value.py | effect R^2 for each attention variant (x exact OV, x program votes) |
| d8_tt_screen.py | corpus pair statistics vs the exact token-token score term TT; SVD of TT |
| d9_fw_votes.py | what function-word / quote / EOS keys really write vs the program's votes |
| d10_family.py | candidate vote tables (family, near, b25, clean next) under exact and rule attention |
| cand_common.py | shared code for the analysis/candidates/recall_*.py files |

Outputs are in analysis/out/candidates/recall/ (JSON files, vote_fit logs, simL/simR/family tables).

## 1. Where the gap is: attention and votes each lose about half (D1)

24 windows. Head-1 shift 0.391 nats. R^2 of head 1's earlier-token effect:

| attention \ votes | exact OV / sdf (weights) | program votes (15 directions, strengths free) | v1 votes (1 direction) |
|---|---|---|---|
| exact head-1 pattern | **0.966** (coef 1.02) | 0.582 | 0.550 (coef **0.92**) |
| RECALL rule (v1) | 0.596 (coef 1.26) | 0.460 | 0.437 (coef **1.57**) |

- The linear direct-logit view (exact attention x OV / sdf) reproduces the ablation effect (0.966), so a linear
  account is enough.
- **The "1.6x stronger" puzzle is attention.** With head 1's real attention, RECALL's own votes fit the head's
  effect with coefficient 0.92. With the rule's attention they need 1.57. The rule spreads attention: the mean
  row entropy is 3.96 nats for the rule and 3.36 for the head (D4). The marginal mass by key group and by
  distance matches almost exactly, so the rule is not too flat on average. It cannot tell *which* keys in a row
  are the right ones, so it hedges, and the averaged votes come out diluted.
- Fixing either half alone gains about 0.12-0.14 R^2. Both together gain 0.5. The errors compound, as the head-1
  card found for output KL.
- Position classes (RECALL v1 R^2 / share of head-1 shift): other 0.45 / 74%; after sentence end 0.58 / 11%;
  word pieces 0.27 / 8%; word onsets 0.25 / 8%. The unexplained part sits mostly at ordinary positions, because
  most of the shift is there. Per position, RECALL is weakest inside and at the start of split words.
- Attended word groups (exact contribution of each group alone, R^2 of the effect): CONTENT_WORDS 0.51,
  FUNCTION_WORDS 0.26, NAMES 0.24, WORD_ENDS 0.08, EOS 0.02, SENTENCE_ENDS 0.00.

## 2. Vote content (D2, D3, D9, D10)

How well RECALL's corpus votes describe each group's exact contribution, under exact attention (D2): NAMES 0.82,
CONTENT_WORDS 0.56, WORD_ENDS 0.48, FUNCTION_WORDS 0.39, SENTENCE_ENDS 0.15. Weighted by group size, the vote-side
residual is mostly content words and function words.

Effect R^2 with extra vote directions (5 per table, one per group), under exact attention / under the rule's
attention:

| votes | exact attention | rule attention |
|---|---|---|
| RECALL 15 directions | 0.582 | 0.460 |
| + family (topic-profile similarity) | 0.608 | 0.466 |
| + near (mean skip-gram PMI d 2..8) | 0.592 | 0.468 |
| + band PMI 25..64 | 0.597 | 0.476 |
| + family + near + band 25..64 | 0.626 | 0.488 |
| NEXT with rare targets zeroed (junk-free) | 0.582 | 0.460 |
| 64-class copy + group topic, next (74) | 0.594 | |
| 64-class copy, topic, band25 + group next (197) | 0.630 | |

What the head really writes (D9, from OV): **pronouns vote for their whole referent family, not just
themselves.** `' she'` writes herself +11.4, she +10.3, her +9.5, She +8.9 and female-name pieces ('aisy',
'olly', ' Molly'), and writes against male and animal nouns. `' he'` writes he, him, himself, He, his, Jack, Jim.
`' they'` writes they, them, their, children. `' I'` and `' you'` write the whole dialogue set (I, you, my, your,
me, closing quotes). RECALL's topic table gives these families only weakly (`' she'` -> herself +2.5, She +2.4)
and spreads noise over rare tokens. Quotes form a dialogue switch: `' "'` writes closing quotes and dialogue words
(+22 to +30). Closing quotes write narrative continuations and strongly *against* dialogue words (-27 to -31).
Sentence ends write sentence starters, as NEXT says. With exact attention their NEXT strength is +1.1, but with
the rule's attention the fitted strength is -0.79. The sign flips because the rule reads sentence ends in the
wrong places.

The "family" table (cosine of topic profiles; closed form from the topic table) is the best single vote addition
under exact attention (+0.026). Under the rule's attention it adds only +0.006, and in the output it gains nothing
(section 4). Once attention is wrong, better vote content barely helps.

## 3. Attention (fit_attn, D4, D5, D6, D7, D8)

Held-out pattern KL(head || rule), all fitted by pattern cross-entropy:

| variant | pattern KL | effect R^2 x exact OV | effect R^2 x program votes |
|---|---|---|---|
| v1 rule | 0.606 | 0.596 | 0.460 |
| + EOS sink | 0.564 | 0.596 | 0.447 (EOS silent 0.459) |
| + key topicality (KL of the topic distribution) | 0.537 | 0.610 | 0.440 |
| + info_8 of key | 0.549 | | |
| + quote flag | 0.588 | | |
| + capitalised flag | 0.601 | | |
| keyflags (cap, unspaced cap, quote, EOS) | 0.536 | 0.590 | 0.444 |
| keyflags + topicality | 0.508 | 0.596 | 0.437 |
| keyflags + topicality + sentence-end pair + pmi1 | 0.477 | 0.613 | 0.447 |
| + pmi1 (k follows q) | 0.601 | 0.606 | 0.467 |
| + log P1(k|q) | 0.606 | 0.593 | 0.459 |
| + E_{v~P1(.|q)} simL(v, k) | 0.606 | 0.595 | 0.460 |
| + sentence end -> earlier sentence end | 0.593 | 0.593 | 0.462 |
| + recency per query group / quadratic recency | 0.606 / 0.606 | 0.595 / 0.595 | 0.461 / 0.460 |
| + exact TT term (weights; insight-only ceiling) | **0.192** | **0.858** | **0.549** |

- **Pattern KL and explanatory value point different ways.** Key-salience terms (topicality, capitals, quotes,
  EOS) cut pattern KL by 21% (0.606 -> 0.477). They raise the effect R^2 with exact votes by at most 0.017, and
  they lower it with program votes. The model really does attend to these keys: the protagonist's name onset
  (`' K'` of Katie at distance 63-126), 'Suddenly', 'Once', ' upon', quotes and EOS. But those keys either write
  little that matters for the next token, or write things RECALL describes badly (quotes; EOS copying itself as
  a FUNCTION_WORD). Pattern cross-entropy weights every key alike. The effect depends on keys whose votes matter.
- **What matters is the token-token term, and the corpus does not supply it.** With head 1's exact token-token
  score term, pattern KL drops to 0.19, and R^2 rises to 0.86 with exact votes and 0.55 with program votes. It
  is essentially the whole attention gap. Its query-key interaction (46% of TT's weighted variance) correlates
  with bigram PMI at r = 0.36 (R^2 0.13) and with the class successor at R^2 0.11. All 13 corpus pair statistics
  together reach 0.20. Low-rank (SVD) smoothing of the bigram PMI does not help (best R^2 0.13 at rank 32). The
  interaction spreads over many components (singular-value shares .33/.15/.11/.085/...). The leading components
  are class-like: after determiners and onsets, attend to noun onsets and nouns rather than suffixes; after a
  sentence end, attend to names; after subject pronouns, attend to reaction and dialogue words ('okay', 'Hi',
  'amazed', 'sorry'). The last of these is not a bigram relation.
- Not sinks in general (D6): quotes are the *loudest* keys (OV std 5.5-7.7 vs about 2.3 typical). EOS and
  ' upon' are ordinary.

## 4. Candidates tested with vote_fit (HEADS = [1])

| candidate | fit/eval windows | paired diff (cand - base) ± SE | head-1 R^2 program -> program+cand (cand alone) | verdict |
|---|---|---|---|---|
| recall_attn_keyflags_t (attention: cap, unspaced cap, quote, EOS, topicality; pattern-fitted) | 16/32 | **+0.0146 ± 0.0012 (worse)** | 0.417 -> 0.344 (0.342) | reject |
| recall_votes_family (5 family directions, v1 attention) | 32/64 | -0.00045 ± 0.00039 | 0.402 -> 0.409 (0.216) | reject |
| recall_attn_TT_exact (INSIGHT-ONLY ceiling: head 1's exact token-token score term in the attention, pattern-fitted) | 24/64 | -0.0321 ± 0.0025 | 0.402 -> 0.499 (0.498) | ceiling, not adoptable |
| recall_votes_b25 (band PMI 25..64 votes per group, v1 attention) | 24/64 | -0.0026 ± 0.0004 | 0.402 -> 0.414 (0.299) | below the 0.005 bar |
| recall_attn_pmi1 (attention + 0.069 * PMI_1(t_i, t_j), pattern-fitted; pattern KL 0.606 -> 0.601) | 24/64 | -0.0046 ± 0.0004 | 0.402 -> 0.363* (0.361) | below the bar (just) |
| **recall_attn_pmi1_votes_b25** (both of the above: 1 attention scalar + 5 vote scalars) | 24/64 | **-0.0069 ± 0.0006** | 0.402 -> 0.373* (0.371) | **passes the KL bar; raises head-1 R^2 when measured correctly (below)** |

\* **vote_fit's head-R^2 artifact for candidates that REPLACE RECALL's directions.** The 'program' R^2 uses
`measure.earlier_votes`, which drops the current token (the diagonal). The candidate's directions come from
`cand.features`, which include the current-token votes (RECALL's softmax runs over j <= i). The harness 'earlier'
ablation leaves the current-token part exact, so those votes are noise for this target and push the candidate's
R^2 down. D12 (16 windows, seed 2000, the same windows vote_fit uses; 15 free directions each) measures every
variant both ways:

| attention | earlier keys only: 15 dirs | + band25 | current token included: 15 dirs | + band25 |
|---|---|---|---|---|
| v1 | 0.437 | 0.459 | 0.368 | 0.384 |
| pmi1 | 0.445 | **0.466** | 0.376 | 0.391 |
| keyflags_t | 0.408 | 0.436 | 0.355 | 0.378 |

So the combined candidate does raise the share of head 1 it explains: 0.437 -> 0.466 (+0.029), with earlier keys
only on both sides. The attention term alone adds +0.008 (D7 on other windows: +0.007), band25 alone +0.021
(D10 with the rule's attention: +0.015). keyflags_t lowers the share on either measure. Suggested fix for
vote_fit (lead's call): compute the candidate's head-R^2 features with the diagonal of RECALL's attention zeroed,
or compare against the 15-direction base under the same convention.

**Recommendation.** `recall_attn_pmi1_votes_b25` is the only candidate that passes: -0.0069 ± 0.0006 nats
(12 SE), and head-1 R^2 0.437 -> 0.466 measured consistently. Each half alone falls just short of 0.005.
Integrating it needs:
1. In head1.attention, add the term `+ PMI1_BONUS * stats.pmi[1][t_i, t_j]`, with PMI1_BONUS = 0.0691. The 9
   attention scalars were refitted jointly on head 1's pattern (fit_attn.json, variant 'pmi1'): MEMORY = 32/0.945
   = 33.86, SALIENCE NAMES 0.745 / SENTENCE_ENDS -0.894 / WORD_ENDS -0.945 / FUNCTION -0.315 / CONTENT -0.042,
   SELF 0.043, SUCCESSOR 0.320. Read it as: prefer earlier words that usually follow the current word (token-level
   bigram PMI) on top of the class-level successor term.
2. A band table in corpus.py, counted exactly like `topic_table` but with distances 25..64: v sits 25..64 tokens
   after the most recent earlier s, first 50M tokens, prior 100. Then one more vote per group:
   BAND[g] * band(s, v), "words of the story's longer-range topic". The screening used the head-1 card's cached
   table, which follows the same rule, so the program's own table should be recounted and the candidate rerun
   before adoption.
3. Refit the vote strengths (the values in analysis/out/candidates/recall_attn_pmi1_votes_b25.json come from
   only 24 fit windows).

## 5. What this says about how head 1 works

1. Head 1 is a recall head whose *selection* is token-specific. Where it looks is mostly a slow recency decay
   plus a query-key preference for "the kind of word that comes next". That preference lives in a rank-32
   token-token term that corpus bigram statistics describe only about 13-20%. Key salience (which words are
   worth remembering: names, topical nouns, story openers, quotes, the end-of-story token) is real and fits
   the pattern, but explains little of what the head *does* to the output.
2. What a recalled word writes is "this word and its family": the token itself, the other forms of the same
   referent (she/her/herself/female names; he/him/his; I/you/my in dialogue) and its topic. For quotes it writes
   a dialogue on/off signal. Sentence ends write sentence starters.
3. The program's RECALL is "too weak by 1.6x" because its attention is diffuse, not because its votes are too
   small. Given the head's real attention, the program's votes have the right strength (coefficient 0.92).
4. Most of the remaining gap is attention, specifically *which* earlier word is picked given the current word.
   Vote refinements only pay off once attention is right: family votes are +0.026 R^2 with exact attention,
   +0.006 with the rule, and 0 in output KL.

## Surprises

- Fitting head 1's pattern better can explain head 1 worse. Key-salience terms cut pattern KL by 21% but lower
  the head-1 R^2 and worsen output KL by +0.015. The v1 end-of-story sink result was an early sign of this.
- RECALL is "1.6x too weak" because of attention diffuseness, not vote size: with head 1's real attention, the
  program's votes fit at coefficient 0.92.
- Head 1 attends to the protagonist's name onset (`' K'` of Katie) from 60 to 126 tokens back, with weight
  0.07-0.21.
- Closing quotes are among the loudest keys: they write strongly *against* dialogue words (-27 to -31 logits).
- vote_fit's head-R^2 for REPLACES candidates includes current-token votes (see the footnote in section 4).

## Dead ends (for logs/dead_ends.md)

- Pattern-fitted `pmi1` attention term alone: -0.0046 ± 0.0004, just below the bar. It only passes together
  with band-25..64 votes.
- Band-25..64 votes alone: -0.0026 ± 0.0004.
- Unclipped class PMI, query-token x key-class PMI, PPMI variants (D11): R^2 0.11-0.13 against the TT interaction,
  the same as the clipped class successor (0.107).

- Key-salience terms for RECALL attention (topicality, capitalised, quote, EOS; pattern-fitted): pattern KL
  0.606 -> 0.508, but output +0.0146 ± 0.0012 worse and head-1 R^2 0.417 -> 0.344. Pattern fidelity on salient
  but unhelpful keys does not transfer to the output.
- Family votes (topic-profile cosine): output -0.0004 ± 0.0004; R^2 +0.007.
- Corpus pair statistics for the token-token attention term: bigram PMI, log P1, positive PMI, reverse bigram,
  PMI_2, successor similarity E_{v~P1}simL, simR, simL, topic in both directions, expected topic votes, and
  rank-4..128 SVDs of PMI_1. Best R^2 0.13 against the exact TT interaction, and +0.007 effect R^2 at most when
  fitted into the rule.
- Recency shape (quadratic; per-query-group decay): no change in pattern KL (0.606).
- Junk-free NEXT (rare targets zeroed): identical R^2. The junk entries on rare tokens do not matter
  (probability-weighted).
