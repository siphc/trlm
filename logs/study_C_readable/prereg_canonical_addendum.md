# Pre-registration addendum (DRAFT): study C part B, readable vs non-readable canonicalisers

DRAFT, 2026-11-06. Written before any part-B output exists: no model of part B has been trained, and the new
measurement (mechanistic convergence) has been run only on the untreated twins attn_s1..s4, for a pipeline check.
Decision: logs/decisions.md item 14 (C part B with both controls, chosen by the human). Design source:
logs/plans/plan_attention_law_and_readability.md, section 3, part B. Part A: logs/study_C_readable/prereg_readable_training.md (FINAL) and
logs/study_C_readable/readable_training_results.md. Everything not stated here is as in part A (configs/side_readable.yaml, rules R1-R7).

**Gates.** G0/G1 (this design, the code and its tests) -> development on seed 5 -> FINAL addendum with the rule-B1
values -> G2 -> confirmatory runs -> results -> G3.

## What has been seen before writing this

- **Part A, confirmatory** (seeds 1-4, lambda 0.3, against the twins attn_s1..s4):
  - Mean pairwise KL among the four seeds: twins 0.417; arm O 0.303 (R = 0.27); arm I 0.310 (R = 0.25).
  - Mean relative cost: O +0.92%, I +1.00%.
  - Convergence ratio: O 0.97, I 3.60.
  - Mean KL to program v1.3: twins 0.554, O 0.398, I 0.515.
  - Arm I's models come closer to each other without coming closer to the program. That is the reason for part B.
- **The behavioural part of the new code has been run on part A's arms.** It reproduces the numbers above. Its
  jackknife standard errors over seeds:
  - R: O 0.010, I 0.072;
  - C: O 0.0015, I 0.0040;
  - log efficiency: O 0.19, I 0.62.
- **Mechanism, twins only:**
  - pairwise attention-pattern KL of the roles LOCAL-0, RECALL and LOCAL-2: 0.92, 0.91 and 1.95;
  - value-code subspace overlap: 0.22, 0.36 and 0.19.
  The same quantities for part A's arms have **not** been computed. They are computed only after the FINAL.
- **Development grid** (seed 0, 100M tokens):
  - arm I at lambda 0.1-0.3 lowered dev CE by 2-3%;
  - arm O at 0.3 cost +0.32%.
  Costs at 100M tokens therefore do not predict one-epoch costs (I at lambda 1: +0.23% at 100M, +3.74% at one
  epoch). See the deviation from the plan in rule B1.

## Question

Pulling four training runs toward a shared target made them about a quarter more alike (part A).
- Is that a property of a **readable** target, or does any shared target do the same?
- At equal cost, which removes more of the variation between runs?
- Does that variation leave the mechanism, or only the behaviour?

## Treatments (all one epoch, seeds 1-4, each against its twin attn_s{k})

| name | level | target | lambda |
|---|---|---|---|
| O_l0.3 (part A, exists) | outputs | program v1.3 | 0.3 |
| I_l0.3 (part A, exists) | heads 0-2: attention and vote tables | program rules and corpus vote tables | 0.3 |
| **O_l1** | outputs | program v1.3 | 1 (pressure curve) |
| **Octl_l<x>** | outputs | attn_s0's next-token distribution | rule B1 |
| **Ictl_l<y>** | heads 0-2: attention and vote tables | attn_s0's heads 0-2: attention and vote (OV) tables | rule B1 |

**The controls are part A's losses with only the target changed** (src/train/readable.py, `readable_target
model:models/final/attn_s0/final.pt`). Everything else is the same as in part A:
- the pool windows and their sampler;
- the coarsening k values (k_out 256; k_h 16, 128, 16);
- the vote-term scale c = 0.836;
- the column and row weights of the vote correlation;
- the 5% ramp, 8 auxiliary windows, the peak LR and the schedule.

Octl's target is attn_s0's top 256 next-token log-probabilities plus the tail mass. Ictl's target is attn_s0's own
heads 0, 1 and 2:
- their attention (top-k_h keys);
- their OV tables, row-standardised in the same metric as the corpus targets.

The program's rules LOCAL-0, RECALL and LOCAL-2 were fitted to exactly these heads. So Ictl's target is the trained
original of arm I's readable approximation. Head 3 and the direct path stay free in both.

**Deviation from the plan: online targets.** The plan proposed a second pool on disk. It would need about 9 GB, and
14 GB are free. Instead, the targets come from a forward pass of the frozen attn_s0 on the same pool windows at every
step. The targets are a pure function of the window, so resuming and determinism are unchanged. The tests confirm:
- lambda 0 is bitwise the plain run;
- resuming is bitwise identical;
- attn_s0 scores 0 on all three losses against itself;
- the attention targets follow the pool's format.

## Development and rule B1 (the controls' lambda)

**Seed 5, not seed 0.** Seed 0 shares its initialisation and batch order with attn_s0, so a seed-0 model pulled
toward attn_s0 is pulled toward its own twin. Its cost and convergence would say nothing about seeds 1-4.

Development runs are on seed 5, which is outside attn_s0..s4, and have one epoch each:
- the lambda = 0 run `none_s5`;
- Octl and Ictl at lambda 0.3 and 1.

**Rule B1.** Each control's lambda is set so that its seed-5 relative cost (dev_full CE against none_s5) equals its
arm's confirmatory mean cost (O +0.92%, I +1.00%):
- Interpolate log lambda linearly between the two run values that bracket the target. Round to two significant
  digits.
- If the target lies outside the costs run so far, run the next grid value. The grid is {0.1, 0.3, 1, 3}; at most
  one extra run per control is expected.
- If the costs are not increasing in lambda, take the run value with the closest cost.
- Code: analysis/canonical/choose_lambda.py, unit-checked on synthetic costs.

**Deviation from the plan.** The plan matched cost at 100M tokens on seed 0. Seed 0 is ruled out above. Costs at
100M do not predict one-epoch costs (see "seen"), and part A's own lambda rule needed a one-epoch check for the same
reason.

## Measurements (analysis/canonical/convergence.py)

All on part A's tools and windows:
- seed_distances.py: train_windows(64, 9100);
- suite.py: align, fidelity, internals, harness;
- the twins' aligned checkpoints.

For treatment X:
- **R** (behavioural convergence) = 1 − D_X / D_twins. D is the mean pairwise KL among the four models (12 ordered
  pairs).
- **C** = mean relative dev_full CE cost against the twins. **E** = R / C. E is not computed if C < 0.25%.
- **Convergence ratio** = R divided by the relative drop of the mean KL to X's own target (the program for O and I,
  attn_s0 for the controls). As in part A.
- **R_P** (attention convergence) = 1 − (mean over roles of pairwise attention-pattern KL) / the twins'. Computed on
  the head-aligned checkpoints, train_windows(16, 9125).
- **R_Q** (value-code convergence) = (Q − Q_twins) / (1 − Q_twins). Q is the mean pairwise subspace overlap of each
  role's value code (uniform token weights).
- **b1** = harness KL to v1.3, against the twins' 0.548. This is (e): does an unreadable canonicaliser also make
  models more describable by the program?
- **Uncertainty:** a leave-one-seed-out jackknife over the four seeds. Each statistic is recomputed on three seeds and
  their three twins. Comparisons are paired by seed.

## Decision rules

**B2, primary: readable vs non-readable, per level** (O_l0.3 vs Octl, I_l0.3 vs Ictl). The two mean costs are
"matched" if they differ by at most 0.4 percentage points.
- If matched: ΔR = R_arm − R_ctl, with its jackknife SE.
  - ΔR ≥ 0.05 and z ≥ 2: "the readable target removes more".
  - ΔR ≤ −0.05 and z ≤ −2: "the non-readable target removes more".
  - Otherwise: "no difference shown".
- If not matched: the efficiency ratio E_arm / E_ctl decides.
  - Ratio ≥ 1.5 and the jackknife z of log(E_arm / E_ctl) ≥ 2: "readable more efficient".
  - Ratio ≤ 1/1.5 and z ≤ −2: "non-readable more efficient".
  - Otherwise: "no difference shown".
  - If either C < 0.25%: not computed.
- **Power, stated now.** With part A's seed spread, the SE of ΔR is about 0.065 at level I and about 0.015 at level
  O. A difference is called only if it is about 0.13 or more at level I, and 0.05 at level O.

**B3, mechanism vs behaviour, per treatment.** R_P and R_Q are each called "converges" if they are ≥ 0.10 with
z ≥ 2. Reading:
- "behaviour converges, mechanism does not": R ≥ 0.15 and neither R_P nor R_Q converges.
- "both converge": R ≥ 0.15 and at least one of R_P, R_Q converges.
- "no convergence": R < 0.15.
For level-I treatments, R_P is partly built in, because attention is trained toward a shared target. R_Q is the
informative measure there, and R_P is the informative one for level-O treatments.

**B4, pressure curve (level O).** Marginal convergence per cost: m = (R(1) − R(0.3)) / (C(1) − C(0.3)).
- "diminishing returns" if m < 0.5 · E(0.3);
- otherwise "no diminishing returns shown".
Reported with R(1), C(1) and b1(1).

**B5, describability.** For each control, b1 against the twins with its SE over seeds. Reported, no threshold.

None of B2-B5 is a verdict on part A. Part A's verdicts stand.

## Expectations (recorded before any part-B output)

- **Level O: the non-readable target removes more per unit of cost.** attn_s0 is itself a member of the set of
  equally good models. The program is outside that set: v1.3 is 0.55 from every seed. So reaching attn_s0 should cost
  less than reaching v1.3.
- **Level I: no difference shown.** Ictl pins heads 0-2 to attn_s0's exact attention and votes, so R_Q should be
  larger than in arm I. Behavioural convergence should be similar, as in arm I, where internals moved and outputs
  hardly did.
- **Pressure curve: diminishing returns.**
- **b1 of the controls: about the twins' (0.54-0.56).** Pulling toward attn_s0 approaches a model that v1.3 is itself
  0.55 from.

## Compute and order

MPS, after study B's queue finishes:
- development: 5 one-epoch runs, about 4-6 h (estimates from part A's step times: plain about 35 min, O-level about
  50 min, I-level about 70 min; the frozen model's forward on 8 windows per step adds a few per cent);
- confirmatory: 12 runs, about 11 h.

CPU: the suite (`--harness`) on each new model and convergence.py.

Queues:
- `scripts/canonical/dev_queue.sh`: refuses to run until a line "G0/G1 approved" is in this file, and waits for study
  B's queue.
- `scripts/canonical/confirm_queue.sh`: refuses to run until the FINAL is committed.

## Verification (G1)

- `src/train/test_sanity_canonical.py`, 6 tests, all pass (2026-11-06):
  - run settings equal to arms O and I apart from the target;
  - lambda 0 bitwise;
  - resume bitwise;
  - the target model scores 0 against itself and attn_s1 does not;
  - attention-target format;
  - target standardisation.
- Part A's `src/train/test_sanity_readable.py` still passes (10 tests).
- convergence.py reproduces part A's published D, C, convergence ratios and b1. The mechanism code gives 0 for a model
  against itself.
- choose_lambda.py: interpolation, the next grid value below and above, the end of the grid, non-monotone costs.

## Not changed

configs/frozen.yaml, harness/, the program, part A's pool, rules and results. The seed-5 development models are
side-study models, not main-track models.
