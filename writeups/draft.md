# What a one-layer TinyStories model computes, how much of it a readable program captures, and why the rest resists

*Running draft, started 2026-11-06; updated at each milestone. Every number comes from a committed results file,
named in brackets; nothing here is new analysis. Not published anywhere. Each section ends with its open caveats.*

**Status of the parts.**

| part | state | results file |
|---|---|---|
| 1. look-follows-write law | done (pre-registered, 1-, 2-, 4-block) | writeups/look_follows_write_predraft.md and its sources |
| 1b. fixed-vote models (study A) | development and exploratory only, no verdicts | logs/fixed_votes_results.md |
| 2. program and coverage | done | logs/coverage.md, logs/mech_v3_report.md |
| 3. the lock | done | logs/token_geometry_results.md, logs/coverage.md |
| 4. readable training (study C part A) | done, both arms GOODHARTED | logs/readable_training_results.md |
| 5. seed convergence (study C part B) | addendum DRAFT, awaiting G0/G1 | logs/prereg_canonical_addendum.md |
| 6. the law under pressure (study B) | done: MIXED, restoring force SLOW; G3 pending | logs/law_pressure_results.md |

## Summary (as of 2026-11-06)

**Models and program.** The models have one attention-only block (4 heads, d_model 128, 2,000-token vocabulary) and
were trained on TinyStories. A readable NumPy program (v1.3) reproduces them from corpus counts and a few hundred
named scalars:
- KL to its target model attn_s0: 0.55 nats per token.
- Coverage: 0.728 of the way from the bigram baseline to the seed floor, the best a description of one seed can be
  expected to reach, measured as the average of the other four seeds (KL 0.256). The value is the same for all five
  seeds.

**Where attention comes from.** A head looks at tokens whose own vote would help predict the next token: the
look-follows-write law. It holds in heads that write to the output and accounts for a third to a half of what exact
attention adds.

**What remains, and the lock.**
- The remaining 0.29 nats are shared by all seeds, so they are not noise.
- They cannot be found in any one part. Making any single head exact inside the program closes at most 0.035 of the
  0.41 nats to the seed consensus. Making everything exact closes 0.13.
- Within one model, attention and vote tables help only together.

**Penalising the law (study B).** Its pair-specific form can be removed at no cost and does not come back, but
attention still goes where the head's own vote helps, as measured by the output test.

**Training the models toward the program.** It does not make the program a faithful description.
- At the output level (arm O), outputs move a quarter of the way to the program; internals do not.
- At the internal level (arm I), heads come to look and vote like the program's rules; outputs do not move.
- The two levels can be bought separately for about 1% cross-entropy.
- Both pressures make training runs about a quarter more alike. Part B tests whether that needs a readable target.

## 1. The look-follows-write law and its limits

**Claim.** The pair-specific part of a head's attention tracks how much that head's own vote from the attended token
would reduce the next-token error relative to a bigram. Details: writeups/look_follows_write_predraft.md
[logs/look_follows_write_results.md, logs/look_follows_write_addendum_results.md, logs/deep_look_write_results.md].
- **One layer.** The law holds in every local head of four held-out seeds:
  - pure-pair correlation 0.34-0.44; another head's vote at most 0.08;
  - it recovers 0.26-0.46 of the exact token-token term's output gain.
- **Training.** Attention and vote co-adapt; usefulness leads only for the pairs that become strongly attended.
- **Deeper models.** The law holds for last-layer heads and is refuted for heads whose output is read by later layers.

**Fixed votes (study A, development only)** [logs/fixed_votes_results.md]. Each head's vote is fixed to a named corpus
table and only attention is learned.
- **Attention adapts to the table.** Changing one head's table from skip2 to skip3 moved its attention peak from
  distance 1 to 2.
- **The law describes a minority of that attention.** It recovers 10-45% of the token-token ceiling. Measuring
  usefulness against what the other heads already predict (explaining away) raises this only for the distance-1 head
  (to 0.50-0.58).
- **No readable model by this route.** A fully readable program built this way is 0.77-0.82 from attn_s0 at CE +11-13%.

*Caveats.*
- Tiny models, one corpus.
- Usefulness is measured with the model's own votes, so it relates two parts of the model; it does not explain them
  without the weights.
- Study A has one seed, and its confirmatory runs were not run (decision 13).

## 2. The program and its coverage

v1.3 [logs/mech_v3_report.md] combines:
- a direct-path bigram (FOLLOW);
- three readable heads: LOCAL-0, RECALL and LOCAL-2, with attention rules that include the law and votes from corpus
  tables;
- head 3 read through skip-gram votes.

It is 0.55 nats from attn_s0 on dev, and 0.54-0.56 from attn_s1..s4 with the same scalars.

**Coverage** (decision 14) [logs/coverage.md] = (KL_bigram − KL_program) / (KL_bigram − KL_floor). The floor is the
probability average of the other four seeds. Values: 0.720-0.737 per seed, mean 0.728 (against the trigram: 0.692).
Against a single sibling seed instead of the average it would read 0.849. That is why the average is the honest floor.

*Caveats.*
- The floor is an estimate from four other seeds. More seeds would lower it a little.
- Coverage is measured on TRAIN windows, not on the test split.

## 3. The lock

**Within attn_s0** [logs/token_geometry_results.md]:
- The model's exact tables help only with its exact attention (KL 0.921 with readable attention), and its exact
  attention helps only with its exact tables (0.523 with corpus tables).
- Both exact: 0.115. The program: 0.551.

**Against the seed consensus** [logs/coverage.md, "Where the remaining gap is"]:
- Exact attention plus exact table for one head inside the program closes 0.035 (RECALL) or at most 0.010 (any other
  head) of the 0.412 nats.
- An exact vote table alone makes things worse for every head.
- Everything exact closes 0.129: three times the sum of the single-head swaps.
- The positions where the consensus's top token appeared earlier (33% + 27% of positions) are mostly common tokens,
  not copies: every seed's RECALL head puts 3-7% of its attention there.

**Reading.** What the seeds share and the program lacks lies in how the parts fit together, not in any part that can
be described on its own. The discovery loop, which improves one component at a time, has no candidate above its
0.005 bar (logs/dead_ends.md).

*Caveats.*
- The oracle swaps refit the program's strengths to attn_s0, but the attention rules are not refitted.
- "Exact" uses attn_s0's weights as data (open track only).

## 4. Output agreement and mechanism agreement come apart under training pressure

**Setup** (study C part A, pre-registered, seeds 1-4, lambda 0.3) [logs/readable_training_results.md]. Each new model
is trained with an extra loss toward v1.3 and compared with its untreated twin.

| | arm O: outputs toward the program | arm I: heads 0-2 toward the rules and corpus tables |
|---|---|---|
| verdict (as amended) | GOODHARTED | GOODHARTED |
| harness KL to v1.3 (twins 0.54-0.56) | 0.39-0.40 | 0.50-0.52 |
| attention-pattern KL to the rules | moves < 2 sigma | 0.59-1.04 -> 0.20-0.27 |
| vote correlation with the tables | +0.02-0.13 | 0.27-0.36 -> 0.69-0.70 |
| cost (dev CE) | +0.92% | +1.00% |

Neither arm shows memorisation, refit failure or (in more than one seed) decorative heads. The verdict reflects
transfer below 0.2 between the levels, not a cheat on the trained measure.

**Reading.** Agreeing with a description at the output level and agreeing with it at the mechanism level are
separately trainable. This is the lock, shown causally in both directions.

*Caveats.*
- With test M at 0.3, as first written, both verdicts are INCONCLUSIVE.
- The positive control showed the out-of-distribution measure (tau_OOD) to be insensitive here; the verdicts rest on
  tau.
- In two arm-I seeds the pattern change fell just under the noise floor.

## 5. Does a shared target remove the arbitrariness between training runs?

**Part A** [logs/readable_training_results.md]: the four treated seeds' mean pairwise KL falls from 0.417 to 0.303
(arm O) and 0.310 (arm I).
- In arm O this matches the movement toward the program (convergence ratio 0.97).
- In arm I the seeds come closer to each other (−26%) while hardly coming closer to the program (−7%; ratio 3.60).

**Part B** [logs/prereg_canonical_addendum.md, DRAFT]. Is this a property of a readable target? Same losses, with
attn_s0 (an unreadable trained model) as the target, at matched cost; a pressure-curve point (arm O at lambda 1);
mechanistic convergence of attention patterns and value codes.
- *Expected (recorded before outputs):* at the output level the unreadable target removes more per unit of cost; at
  the internal level no difference is shown.
- *Power:* differences below about 0.13 (level I) or 0.05 (level O) in convergence will not be called.

*Caveats.* Four seeds per arm; a jackknife over seeds.

## 6. The law under pressure (study B)

Pre-registered [logs/prereg_law_pressure.md; results logs/law_pressure_results.md]. Seeds 1-4 are trained with a
penalty on the in-batch correlation between attention scores and the usefulness of the head's own vote (lambda 1),
each against an unpenalised twin, then both are continued for 3,000 steps without the penalty.
- **Verdict MIXED.** Cells: FREE BREAK (seed 1), PARTLY BROKEN at low cost (seeds 2 and 4), INCONCLUSIVE (seed 3, whose
  penalised statistic fell only to 28% of the twin's).
- **The cost is LOW in every seed** (-0.13% to +0.39% cross-entropy).
- **The law's strength** (pure-pair correlation) falls from 0.33-0.34 to 0.06-0.21; in RECALL heads to almost nothing.
- **Restoring force SLOW.** Recovery is 0.05-0.14 of what was removed; the twins do not move.
- **The two held-out measures come apart.** The own-usefulness output gain stays at 0.90-1.03 of the twin's, with
  unchanged fitted coefficients.

**Reading.** The pair-specific form of the law is not load-bearing for prediction: it can be removed for free and
stays removed. The coarser statement, that attention goes where the head's own vote helps, survives as measured by the
output test.

*Caveats.*
- One lambda; four seeds, one of them failing the manipulation check; MPS only.
- The continuation is short and at a low learning rate.
- Which part of usefulness carries the surviving gain is an exploratory question (logs/law_pressure_results.md).

## Open items for a real draft

- Figures:
  - coverage bar (bigram, trigram, program, floor);
  - lock factorial;
  - arm O and arm I movement at the two levels;
  - pairwise seed distances.
- Decide the headline. Candidates: "readable descriptions hit a whole-model lock", or "output and mechanism agreement
  are separately trainable".
- Replicate one result outside these models (a public TinyStories model).
- Related work: check claims against the full papers (writeups/look_follows_write_predraft.md, references).
