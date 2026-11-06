# Results: can the look-follows-write law be Goodharted? (study B, confirmatory runs)

Written 2026-11-06 from analysis/out/law_pressure/verdicts.json (analysis/law_pressure/verdicts.py), measures.json (the
in-batch correlations) and report.json (analysis/law_pressure/report.py, per-head numbers, reported not tested).
Pre-registration: logs/prereg_law_pressure.md (FINAL, 2026-10-20). Running log: logs/law_pressure.md. Training log
logs/runs/law_pressure_confirm.out; law tools logs/runs/law_pressure_confirm_lawtools.out and
law_pressure_lawtools_rest_{a,b,c}.out. Gate G3 (the human's review) is pending.

**Assumptions stated up front.**
- All 16 models (twin, penalised, and both continued, for seeds 1-4) were trained on the laptop GPU (MPS), the
  fallback hardware the FINAL names (decision 13). So every paired comparison is within one kind of hardware.
- lambda* = 1, law_windows = 16, arm both only (rules R1-R3, fixed in development on seed 0).
- The law's strength is the mean over the four heads of the pure-pair correlation at d = 1 with the head's own
  usefulness. The own gain is the mean over heads of the own-usefulness arm's output gain. Both come from goal 1's law
  tools, which the penalty never sees.
- No test-split looks. Exploratory analyses are labelled as such and carry no verdict weight.

## Answer in brief

**Verdict for arm both: MIXED. Restoring force: SLOW.**

- **The penalty took hold** in 3 of 4 seeds. The trained statistic, the in-batch correlation, fell to 7-15% of the
  twin's. Seed 3 reached 28%, which fails the manipulation check (at most 20%).
- **The law's strength fell in every seed,** from 0.33-0.34 to 0.06-0.21. That gives BROKEN in seeds 1 and 3 and
  PARTLY BROKEN in seeds 2 and 4. The fall comes almost entirely from the pure-pair correlation.
- **The law's other held-out measure did not fall.** The own-usefulness output gain stayed at 0.90-1.03 of the twin's.
  The fitted usefulness coefficients barely moved.
- **The cost was LOW in every seed:** -0.13% to +0.39% dev_full cross-entropy.
- **The law hardly came back.** After 3,000 steps without the penalty, recovery was 0.05-0.14 of what had been removed.
  The twins, continued the same way, did not move (0.323-0.340).

So the two held-out measures of the law, which agree in normal models, come apart under this pressure. The
pair-specific part of attention stops tracking own usefulness, at no measurable cost. The part of the law that buys
output (attention given to keys whose own vote helps) is unchanged. The development run on seed 0 already showed this
pattern (law_pressure.md: own gain at 77-89% of the control). The confirmatory runs show it more strongly.

## Per seed (arm both, lambda 1, against the lambda = 0 twin of the same seed)

| seed | in-batch corr twin -> penalised (ratio; M) | law strength twin -> penalised | own gain twin -> penalised (ratio) | dev_full twin -> penalised (cost) | law | cell |
|---|---|---|---|---|---|---|
| 1 | 0.141 -> 0.015 (0.11; pass) | 0.325 -> 0.099 | 0.077 -> 0.079 (1.03) | 2.5086 -> 2.5159 (+0.29%) | BROKEN | FREE BREAK |
| 2 | 0.125 -> 0.018 (0.15; pass) | 0.326 -> 0.126 | 0.078 -> 0.073 (0.94) | 2.5401 -> 2.5369 (-0.13%) | PARTLY BROKEN | PARTLY BROKEN, cost LOW |
| 3 | 0.133 -> 0.037 (0.28; **fail**) | 0.330 -> 0.056 | 0.070 -> 0.064 (0.91) | 2.4917 -> 2.5013 (+0.39%) | BROKEN | INCONCLUSIVE |
| 4 | 0.131 -> 0.009 (0.07; pass) | 0.343 -> 0.213 | 0.090 -> 0.081 (0.90) | 2.5071 -> 2.5098 (+0.11%) | PARTLY BROKEN | PARTLY BROKEN, cost LOW |

- **Per-seed rule.** BROKEN means strength < 0.10 or own gain at most 0.25 of the twin's. INTACT means strength at
  least 0.25 and own gain at least 0.5 of the twin's.
- **Every state is decided by strength.** The own-gain condition is met as for an intact law in all four seeds.
- **No cell holds in 3 of 4 seeds,** so the verdict is MIXED. Seed 3 is INCONCLUSIVE and counts toward no cell.
- **Seed 3** has the weakest suppression of the trained statistic but the lowest held-out strength (0.056). How
  strongly the penalised statistic is suppressed does not order the seeds by how much of the law breaks.

## Restoring force (3,000 steps, 49M tokens, without the penalty)

| seed | strength: penalised -> continued | twin continued | recovery | own gain: penalised -> continued |
|---|---|---|---|---|
| 1 | 0.099 -> 0.110 | 0.323 | 0.05 | 0.079 -> 0.081 |
| 2 | 0.126 -> 0.155 | 0.323 | 0.14 | 0.073 -> 0.082 |
| 3 | 0.056 -> 0.088 | 0.327 | 0.12 | 0.064 -> 0.066 |
| 4 | 0.213 -> 0.220 | 0.340 | 0.06 | 0.081 -> 0.084 |

- **Rule.** FAST means recovery of at least 0.5 in 3 of 4 seeds; NONE means at most 0.1 in 3 of 4.
- **Verdict SLOW.** No seed reaches 0.5 and two reach at most 0.1 (seeds 1 and 4). Seeds 2 and 3 sit just above 0.1
  (0.14 and 0.12).
- **The continuation's own effect is nil.** The twins' strength changes by at most 0.003.
- **Reading:** the low-correlation state is stable under ordinary training at a low learning rate (2e-4 to 2e-5).
  This continuation does not test whether the law would return under a full learning-rate schedule.

## Where the work goes (reported, not tested; report.json)

Means over 8 heads of each kind; the kind is the twin's head classification. One head changes kind under the
penalty: seed 2, head 0, RECALL -> LOCAL.

| kind | model | pure-pair corr, own | own gain | token-token ceiling gain | max abs other head's corr | base-rule pattern CE |
|---|---|---|---|---|---|---|
| LOCAL | twin | 0.393 | 0.103 | 0.073 | 0.095 | 2.079 |
| LOCAL | penalised | 0.204 | 0.095 | 0.063 | 0.054 | 2.121 |
| RECALL | twin | 0.269 | 0.055 | 0.113 | 0.055 | 3.553 |
| RECALL | penalised | 0.044 | 0.053 | 0.141 | 0.072 | 3.538 |

- **The pure-pair correlation falls by half in LOCAL heads and to almost nothing in RECALL heads.** In three RECALL
  heads it turns negative (-0.09, -0.11, -0.19).
- **The own-usefulness coefficients barely change.** For example seed 1, head 0: c1, c2 = 0.26, 0.27 -> 0.27, 0.29.
  Head by head they are in analysis/out/look_write/law_pressure_*/arms.json. The output gain also barely changes.
- **Head-specificity does not weaken** in the confirmatory seeds: the largest other-head correlation is 0.14 before
  and after. In development, seed 0 at lambda 1-3 reached 0.23.
- **Position and spelling explain the pattern about as well as before.** The base-rule pattern CE moves by 0.015 (RECALL)
  and 0.042 nats (LOCAL), means by kind.
- **The token-token ceiling,** what exact pair attention adds over the base rule, stays at 0.06-0.14.

## Recorded expectations against the outcome

| recorded before outputs | outcome |
|---|---|
| DRAFT: the in-batch correlation is easy to suppress | yes in 3 of 4 seeds (seed 3 reaches 28%, just above the 20% check) |
| DRAFT: the aggregate law weakens but does not vanish, at a cost of 1-3% | it weakens (strength 0.06-0.21) at a cost below 0.4%. Cost is lower than expected |
| DRAFT: a FAST return without the penalty | wrong: SLOW (recovery 0.05-0.14) |
| after development: PARTLY BROKEN with LOW cost in most seeds, verdict MIXED | verdict MIXED as expected. PARTLY BROKEN, LOW in 2 of 4; the others FREE BREAK and INCONCLUSIVE |

## What this does and does not show

- **The law's pair-specific form is not load-bearing for prediction.** Under the pre-registered definitions it can be
  removed or halved for free (cost below 0.4%), and it does not come back on its own. In the sense the study asked
  about, it is not an interpretability handle that resists training pressure.
- **But "attention goes where the head's own vote helps" survives, measured by the output test.** The own-usefulness
  arm keeps its gain and its coefficient. The penalty is a pooled instance-level correlation, double-centred within
  each query's seven keys. It appears to remove the part of the law that varies with the particular (key, query)
  pair, and leave the part that the output test uses.
- **One explanation was tested and refuted** (exploratory): token main effects do not carry the surviving part; they
  lose their link to usefulness too. Which part of attention carries it is open (the tail of strongly attended pairs,
  distances 2-7, or position).
- **Limits.**
  - One lambda (1).
  - Tiny one-layer models, four seeds, MPS only.
  - Seed 3 fails the manipulation check.
  - The continuation is short and at a low learning rate.

## Exploratory (not pre-registered, written after the verdicts)

*Main effects vs pair-specific part* (analysis/law_pressure/main_effects.py; per model
analysis/out/law_pressure/main_effects/<run>.json; logs logs/runs/law_pressure_main_effects_*.out).

**Hypothesis:** the surviving output gain is carried by token main effects (keys whose vote is useful whatever the
query), which the pure-pair correlation removes by double-centring.

**Method.** On the same d = 1 type table (key, query cells with N_1 >= 20, weighted by N_1), the exact score and own
usefulness are each split into an additive part (a_k + b_q, weighted least squares) and a residual. Four weighted
correlations:
- raw;
- additive part with additive part;
- key effects with key effects, over keys weighted by count;
- residual with residual (the pre-registered pure-pair correlation, reproduced).

The key effect is the weighted mean of the additive fit over the key's cells. So it includes a count-weighted mean of
the query effects; it is an approximation of a_k.

Mean over the 8 heads of each kind (the twin's classification):

| kind | model | raw | additive | key effects | pure-pair | additive share of score variance |
|---|---|---|---|---|---|---|
| LOCAL | twin | 0.408 | 0.459 | 0.435 | 0.393 | 0.80 |
| LOCAL | penalised | 0.095 | 0.081 | 0.036 | 0.204 | 0.91 |
| RECALL | twin | 0.285 | 0.318 | 0.342 | 0.269 | 0.80 |
| RECALL | penalised | 0.023 | 0.020 | -0.095 | 0.044 | 0.84 |

**Refuted.**
- The main-effect parts lose their link to usefulness even more completely than the pair-specific part.
- In the penalised models the key effects are uncorrelated with usefulness, or negatively correlated (seed means
  0.003, -0.077, -0.020, -0.023, against 0.29-0.45 in the twins).
- So on the exact d = 1 type table the scores no longer track own usefulness in any part, yet the own-usefulness arm,
  fitted to attention patterns on real windows, keeps its coefficient and its output gain.

**Remaining candidates (untested):**
- **The tail.** The arm is fitted by pattern cross-entropy, which weights the pairs a head actually attends to. The
  type-table correlations weight every frequent pair by its count. The penalty, a correlation pooled over all
  instances, could remove the link on the bulk of pairs while keeping high scores on the few high-usefulness pairs
  that carry the attention. That fits goal 1's finding that usefulness leads only for strongly attended pairs.
- **Distances 2-7.** The arm's c2 feature uses them; the type-table analysis covers d = 1 only.
- **Position.** The type table averages scores over query positions.

These would be next only at the human's request. This line has used its exploratory step (logs/dead_ends.md).
