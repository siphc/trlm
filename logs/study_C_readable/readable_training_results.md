# Results: can a language model be trained to be readable? (side study, confirmatory runs)

Written 2026-11-06 from analysis/out/readable_training/verdicts.json and seed_distances.json (verdicts.py and
seed_distances.py, run by scripts/readable/finish.sh, log logs/runs/readable_verdicts.out). Pre-registration:
logs/study_C_readable/prereg_readable_training.md (FINAL, plus the amendment of test M, decision 11); clarifications 1-9 in
logs/study_C_readable/readable_training.md, phase 4, all written before any confirmatory output. Gate G3 (the human's review) is pending.

**Assumptions stated up front.** The target is program v1.3, frozen. Each new model is compared with its untreated
twin attn_s{k} (same initialisation and batch order). sigma_T is the spread of tool T over attn_s1..s4 (rule R4). All
numbers below come from the suite and the harness; no test-split looks.

## Answer in brief

Both arms get the pre-registered verdict **GOODHARTED**. That holds with test M at 0.2 (the amended value) and with the
literal noise floor; with M at 0.3 as first written, both arms are INCONCLUSIVE. GOODHARTED is reached the same way in
both arms: the transfer tau is at most 0.2 in 3 of 4 seeds. Agreement with the program at the level that was trained
does not carry over to the other level.

- **Arm O (train the outputs toward the program):** the model's outputs move a quarter of the way to the program
  (harness KL 0.54-0.56 -> 0.39-0.40) and stay there out of distribution and after refitting. But its heads do not look
  or vote more like the program's rules: the pattern KL to the rules moves by less than 2 sigma in every seed.
- **Arm I (train attention and votes toward the program):** the heads now look like the rules (pattern KL 0.59-1.04 ->
  0.20-0.27) and vote like the tables (weighted vote correlation 0.27-0.36 -> 0.69-0.70), and this holds out of
  distribution. But the outputs stay almost as far from the program (harness KL 0.54-0.56 -> 0.50-0.52).
- Neither arm shows memorisation, refit failure or, in more than one seed, decorative heads. So this is not Goodhart
  in the sense of a cheat on the trained tool. What the verdict reflects is that **output agreement and mechanism
  agreement come apart**: either one can be bought for about 1% cross-entropy without the other.

## Verdicts

| arm | verdict (M 0.2) | M 0.3 as first written | literal noise floor | how |
|---|---|---|---|---|
| O, lambda 0.3 | GOODHARTED | INCONCLUSIVE (M fails in 4 of 4) | GOODHARTED | tau <= 0.2 in 3 of 4 seeds |
| I, lambda 0.3 | GOODHARTED | INCONCLUSIVE (M fails in 2 of 4) | GOODHARTED | tau <= 0.2 in 3 of 4 seeds |

In both arms, test M passes in every seed, no seed reaches tau >= 0.5 and tau_OOD >= 0.5 together, tau_OOD <= 0.2 in no
seed, and each flag occurs in at most 1 seed.

### Arm O per seed

Trained tool: harness KL (b1). Untrained tool: the mean of the pattern-KL and the uniform vote-correlation closures.

| seed | trained closure | untrained closure | tau | tau_OOD | flags | cost | b1 twin -> new | b2 (refit) | lock drop |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.27 | 0.00 | 0.00 | 1.18 | none | +1.23% | 0.547 -> 0.399 | 0.409 | +19% |
| 2 | 0.28 | 0.04 | 0.13 | 0.90 | none | +0.56% | 0.561 -> 0.403 | 0.412 | +17% |
| 3 | 0.28 | 0.06 | 0.21 | 0.98 | ii | +1.08% | 0.540 -> 0.388 | 0.387 | +31% |
| 4 | 0.29 | 0.00 | 0.00 | 0.94 | none | +0.81% | 0.545 -> 0.389 | 0.392 | +22% |

- The pattern KL to the rules barely moves: 0.80 -> 0.75, 1.04 -> 0.77, 0.59 -> 0.49 and 0.65 -> 0.50. Each change is
  below the 0.40 floor (2 sigma). The vote correlation rises by 0.02-0.13.
- Out of distribution the output agreement holds: tau_OOD is 0.90-1.18. On the program's own samples the KL falls
  most (for example 1.24 -> 0.67 in seed 1).
- Seed 3's flag (ii): head 2's ablation shift is 0.12 of its twin's and head 3's is 2.7 times. The twin attn_s3 has an
  unusually large head-2 shift (3.5 against 0.9-1.2 in the other twins).

### Arm I per seed

Trained tool: the mean of the pattern-KL and the weighted vote-correlation closures. Untrained tool: harness KL (b1).

| seed | trained closure | untrained closure | tau | tau_OOD | flags | cost | pattern KL twin -> new | vote corr. (weighted) | b1 twin -> new | lock drop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.61 | 0.05 | 0.08 | 0.96 | none | +1.29% | 0.80 -> 0.26 | 0.32 -> 0.69 | 0.547 -> 0.521 | -23% |
| 2 | 0.66 | 0.08 | 0.12 | 0.98 | none | -0.05% | 1.04 -> 0.26 | 0.27 -> 0.69 | 0.561 -> 0.515 | -19% |
| 3 | 0.29 | 0.05 | 0.16 | undefined | ii | +1.84% | 0.59 -> 0.20 | 0.28 -> 0.70 | 0.540 -> 0.515 | +5% |
| 4 | 0.26 | 0.08 | 0.28 | undefined | none | +0.90% | 0.65 -> 0.27 | 0.36 -> 0.70 | 0.545 -> 0.504 | -5% |

- **The noise-floor risk recorded in clarification 9 happened.** In seeds 3 and 4 the pattern KL fell by 0.385 and
  0.385 nats, just under the 0.40 floor. Their pattern closure therefore counts as 0, which halves the trained closure
  and leaves tau_OOD undefined. All four arm-I models end at nearly the same pattern KL (0.20-0.27) whatever their twin
  started from. Read without the floor, seeds 3 and 4 behave like seeds 1 and 2. This changes neither verdict: with the full
  closure in the denominator, tau would be 0.08 (seed 3) and 0.14 (seed 4).
- Work does not leave the constrained heads. Ablating head 0 costs 1.6-3.1 times what it costs in the twin, and head
  3's shift falls to 0.21-0.93 of its twin's. Seed 3's flag (ii) is head 2 at 0.37 of the twin (again attn_s3's
  unusual head 2).

## Reported separately (no verdict weight)

| | arm O | arm I |
|---|---|---|
| cost, dev_full CE (mean +- SE, 4 seeds) | +0.023 +- 0.004 nats (+0.92%) | +0.025 +- 0.010 nats (+1.00%) |
| b1 harness KL to v1.3 (twins 0.540-0.561) | 0.388-0.403 | 0.504-0.521 |
| b2 after refitting the program's scalars | 0.387-0.412 (refit gains nothing) | 0.505-0.525 |
| lock (d2), LOOSENED if it drops by >= 30% | drops 17-31%; LOOSENED in 1 seed | tighter in seeds 1-2 (+19-23%), about unchanged in 3-4 |
| mean pairwise KL among the 4 seeds (twins 0.417) | 0.303 (-27%) | 0.310 (-26%) |
| mean KL to the program on the same windows (twins 0.554) | 0.398 (-28%) | 0.515 (-7%) |
| seed convergence ratio (>= 0.5 converge) | 0.97 (converge) | 3.60 (converge) |
| attn_s0 specificity, as worded (expected <= 0.02) | 0.013-0.049 | -0.012 to -0.008 |
| attn_s0 specificity without the twin | -0.020 to 0.003 | -0.021 to -0.006 |
| harness claim tests passed, of 7 (twins s1..s4: 6, 4, 6, 7) | 6, 5, 6, 6 | 7 in every seed |

- **Seeds become more alike in both arms, by about a quarter.** In arm O this matches how far the models moved toward
  the program. In arm I the models come closer to each other (-26%) while hardly coming closer to the program (-7%).
  Pushing the internals toward the program's rules removes part of the arbitrariness between training runs without
  making the output program-like. This bears directly on study C ("canonical"), whose part A these runs are.
- **The pull toward v1.3 is not a pull toward attn_s0.** The program was fitted to attn_s0, but the new models are no
  closer to attn_s0 than to the other attn seeds (specificity without the twin -0.02 to 0.003).
- **Arm I's lock tightens.** In seeds 1-2, swapping in the program's attention and table for a head costs more than in
  the twin, even though the heads now look like the rules. Pattern KL and vote correlation are close to the program's,
  but the parts they do not measure are what the output depends on.

## Positive control: the suite is partly insensitive

The control (arm O, lambda 10, a pool of only 256 windows, seed 0, 100M tokens) was meant to show that the suite catches
a model that games a small pool. Against its lambda = 0 twin (clarification 8):

- **Flag (i) works.** The memorisation gap is 0.158 nats against the 0.02 threshold.
- **tau_OOD does not.** The control's agreement transfers out of distribution as well as on dev (tau_OOD 1.20; b1
  closure 0.35, OOD closures 0.39-0.47). Against attn_s0 instead, tau_OOD is 1.29.

The pre-registration requires both, so **the suite is declared insensitive** in the sense the rule defines. The
practical meaning: on the dev split and the four out-of-distribution sets, the program's outputs are similar enough
that agreement learned on 256 windows already generalises. tau_OOD cannot tell memorisation from transfer in this
setup; only the pool gap (flag i) can. Both arms' verdicts rest on tau, not on tau_OOD, so this does not change them.
It does mean that "tau_OOD about 1" in arm O is weak evidence that the output agreement is genuine.

## Against the recorded expectations

Expected: "arm O closes much of b1 at little cost with partial internal transfer; arm I matches attention easily,
transfers little to the output, and risks decorative heads at high lambda." Arm I came out as expected (no decorative
heads at lambda 0.3). For arm O, b1 closed by 0.27-0.29 ("much" is generous) and there was no internal transfer.

## What this means for the project

- Training a model toward v1.3 does not make v1.3 a faithful description of it. The best case, arm O, leaves the
  program 0.39 nats away. That is about the distance between two ordinary seeds (0.38-0.42) and costs 1% CE.
- The output-level and mechanism-level descriptions of these models are separately trainable. This is the same
  decoupling the main track met as "the lock", now shown causally in both directions: matching one level does not move
  the other.
- Arm I's seed convergence is the one positive handle: internal pressure makes training runs less arbitrary. Study C
  part B (a pressure curve and a non-readable canonicaliser control) would test whether that is about readability or
  about any shared internal target.

## Files

- Verdicts and every tool value: analysis/out/readable_training/verdicts.json; seed distances: seed_distances.json.
- Suites: analysis/out/readable_training/readable_{O,I}_l0.3_s{1..4}/suite.json; twins attn_s1..s4; positive control
  dev_O_l10_pool256_s0_100M and its twin dev_none_s0_100M.
- Models: models/readable/{O,I}_l0.3_s{1..4} (gitignored).
