# Pre-registration (FINAL): can the look-follows-write law be Goodharted? (study B, `law_pressure`)

FINAL, 2026-10-20. The DRAFT was committed 2026-10-18 (b1d0897) before any model was trained with the penalty. This
version adds the values fixed by the decision rules (development outputs only, seed 0), what has been seen, and the
deviations. It is committed before any confirmatory model (seeds 1-4) is trained; the confirmatory runs wait for the
human's gate G2. Decisions: logs/decisions.md items 10 and 12. Plan: logs/plan_attention_law_and_readability.md
section 5. Code: src/train/law_pressure.py (penalty), src/train/train.py (hook, default off), src/train/runs.py
(`law_dev`, `law`, `law_restore`), src/train/test_sanity_law_pressure.py (11 tests), analysis/law_pressure/rules.py,
analysis/law_pressure/verdicts.py (the outcomes below; refuses to run before this commit). Running log:
logs/law_pressure.md.

## What has been seen before writing the DRAFT

- The law in normal models (logs/look_follows_write_results.md and addendum): per local head, the pair-specific part of
  attention follows the usefulness of the head's own vote, aggregated over the corpus (pure-pair correlation 0.34-0.44,
  other heads' at most 0.08; own usefulness gains 0.06-0.18 nats in the output test); about 0.24 attention score per
  logit of usefulness. Averaged over all four heads (as the law strength below): 0.326-0.351 in attn_s0..s4, mean
  own-usefulness gain 0.070-0.090.
- The penalty's trained quantity in the normal models: the pooled in-batch correlation of attention scores and
  instance usefulness (definition below) on train_windows(16, 9300) is positive for every head of attn_s0..s4:
  0.106-0.161 (attn_s0: 0.131, 0.143, 0.132, 0.110).
- Sanity tests pass on CPU (terms equal a direct computation; arms route gradients as intended; lambda = 0 bitwise the
  plain run; bitwise resume; optimising the penalty alone on one batch of attn_s1 removes most of the correlation).

## Values fixed by the rules (analysis/out/law_pressure/rules.json)

| rule | value | from |
|---|---|---|
| R1 | **lambda* = 1** | development grid: mean abs in-batch correlation 0.110 at lambda 0; 0.018 (16%) at 1, 0.011 at 3, 0.006 at 10, 0.005 at 30, 0.004 at 100 |
| R2 | **law_windows = 16** (in every run of the study) | laptop GPU: plain 0.096 s/step, penalised 0.247 s/step (2.56x > 1.6x) |
| R3 | **neither arm attn nor arm vote admitted** | at lambda 1: arm attn 0.056 (51% of control), arm vote 0.080 (72%) |
| R4 | defaults kept: law measurements on goal 1's windows (32 fit, 32 evaluation); continuation analysed at its end | |

**Confirmatory runs:** for seeds 1-4, the twin (`law --arm both --lam 0 --law_windows 16`) and the penalised model
(`law --arm both --lam 1 --law_windows 16`), one epoch each; both continued without the penalty (`law_restore`); goal 1's
law tools on all 16 models; in-batch correlations. All on the cloud instance (NVIDIA L4) if it is available, otherwise
all on the laptop GPU.

## What has been seen in development (seed 0, 100M tokens, laptop GPU; no verdict weight)

| lambda | in-batch corr (share of control) | law strength: pure-pair corr, mean of 4 heads | own-usefulness gain (mean) | dev_full CE (vs control) |
|---|---|---|---|---|
| 0 (control) | 0.110 (1.00) | 0.320 | 0.057 | 2.6618 |
| 1 | 0.018 (0.16) | 0.175 | 0.048 | +0.22% |
| 3 | 0.011 (0.10) | 0.189 | 0.051 | -0.35% |
| 10 | 0.006 (0.05) | 0.142 | 0.048 | -0.59% |
| 30 | 0.005 (0.05) | 0.140 | 0.044 | +0.08% |
| 100 | 0.004 (0.035) | 0.256 | 0.015 | +0.81% |
| arm attn, 1 | 0.056 (0.51) | 0.179 | 0.050 | +0.34% |

The penalised statistic falls 6-30x at no measurable cost; the law strength roughly halves at lambda 1-30 and partly
returns at 100, where the own-usefulness gain falls to a quarter: the two held-out measures of the law come apart
under strong pressure. Head-specificity weakens at lambda 1-3 (another head's usefulness up to |0.23| pure-pair
correlation; control <= 0.08). On the definitions below, seed 0 at lambda* would be PARTLY BROKEN with LOW cost.

## Deviations and clarifications (all before any confirmatory output)

- Hardware (decision 12): development on the laptop GPU; confirmatory runs, twins and continuations on one hardware.
- The twin of a penalised model is the lambda = 0 run of the same seed on the same hardware (DRAFT text edited before
  any output; attn_s{s} are on the laptop).
- R2's scope (every run uses the development value) was written after R2's timing was measured; it does not depend on
  the measured value.
- The outcome thresholds below are unchanged from the DRAFT although the development results are known.
- Counting: a seed failing M is INCONCLUSIVE and counts toward no cell; the verdict names a cell only if that same cell
  holds in at least 3 of 4 seeds; a recovery with a denominator <= 0 is undefined and counts toward neither FAST nor NONE.
- "Where the work goes" is reported, not tested: goal 1's base-rule pattern cross-entropy against the pair arms, and the
  other heads' usefulness arms (head-specificity).

## Question

Training against probes is known to be evaded cheaply. The look-follows-write law is not a probe: it follows from
first-order training dynamics. If training penalises it, does the model (a) pay in prediction quality, (b) break the
law for free, or (c) suppress the penalised measurement while the law, measured in ways the penalty never sees, stays?
A law that is costly to break would be an interpretability handle that resists training pressure.

## Penalty (src/train/law_pressure.py)

For each head h and each pair (query i, key j = i - d), d = 1..7, queries 7..255 of each training window:
s_ij = q_i . k_j / sqrt(32) (the raw score); u_ij = OV_h(t_j, y_i) - sum_v P1(v | t_i) OV_h(t_j, v), the usefulness of
the head's own vote from t_j for the actual next token y_i beyond the bigram (OV_h through the direct path, the final
LayerNorm gain, scale-free; P1 FOLLOW's bigram rule). Both double-centred (each query's 7 keys, then each distance);
penalty = lambda x mean over heads of corr(s, u)^2, pooled over the batch (64 windows). lambda ramps linearly over the
first 5% of steps. Arms: **both** (gradients reach attention and votes), **attn** (u detached), **vote** (s detached).

## Models

TinyLM attention-only, configs/frozen.yaml v3, one epoch (527,482,880 tokens), peak LR 2e-3, fp32. The penalty
consumes no random numbers, so seed s has exactly the initialisation and batch order of attn_s{s}.
- Hardware (decision 12): each comparison within one kind of hardware. The development grid and its lambda = 0 control
  on one (the laptop GPU, MPS, if it runs before the instance is available); the confirmatory runs, their twins and
  the continuations all on the cloud instance (NVIDIA L4, CUDA) if available, otherwise all on the laptop GPU. The **twin** of a penalised model is the lambda = 0 run of the same seed
  on the same hardware (`law --arm both --lam 0`: initialisation and batch order of attn_s{s}, and bitwise the plain
  run on that hardware), so hardware differences cannot enter the paired comparisons.
- Development (seed 0, 100M tokens): arm both at lambda in {0, 1, 3, 10, 30, 100} (lambda 0 is the control); arms attn
  and vote at lambda* (R1).
- Confirmatory: twins (lambda 0) and arm both at lambda*, seeds 1-4; arms attn and vote at lambda*, seeds 1-4, if R3
  admits them.
- Restoring force: each confirmatory model and its twin continued without the penalty for 3,000 steps (49M tokens;
  fresh AdamW, 1% warmup, cosine from 2e-4 to 2e-5; batch order of seed s + 10).

## Decision rules (development only; values in the table above)

- R1: lambda* = the smallest grid value whose 100M-token run (arm both) has a mean over heads of |in-batch correlation|
  (on train_windows(16, 9300), src/train/law_pressure.corr_on) at most 20% of the control's; the largest grid value if
  none does. No cost cap: the cost is an outcome.
- R2: if a penalised step takes more than 1.6x the plain step on the development hardware, the penalty uses the first
  16 windows of each batch (TrainConfig.law_windows), in every run of the study (lambda* is chosen with that penalty).
- R3: arms attn and vote enter the confirmatory runs only if their 100M-token run at lambda* meets R1's 20% criterion.
- R4: development may fix only these analysis details: the number of windows of the law measurements (default goal 1's:
  32 fit, 32 evaluation), and the restoring-force checkpoints analysed (default: the end of the continuation).

## Measurements (per model)

- **Trained quantity:** in-batch correlation per head (R1's windows).
- **Cost:** dev_full CE, paired against the twin.
- **Held-out law** (goal 1's tools, which the penalty never sees; analysis/look_write/uov_tables.py and fit_arms.py,
  generalised to checkpoint paths before use): corpus-aggregated usefulness tables UOV_h,d; per head the own-UOV arm's
  output gain over the base rule; the pure-pair correlation at d = 1 with own UOV; the other heads' and the shuffled
  controls; the fitted coefficient. The law's strength of a model = mean over heads of the pure-pair correlation with
  own UOV (normal models 0.34-0.44 for local heads).
- **Where the work goes:** per-head mean-ablation shifts; the share of attention explained by the base rule (position and
  spelling) vs pair features; head-specificity (other heads' usefulness predicting this head's attention).

## Outcomes per confirmatory seed (arm both)

- **M (manipulation check):** mean |in-batch correlation| at most 20% of the twin's; otherwise the seed is INCONCLUSIVE.
- Held-out law **INTACT**: mean pure-pair correlation >= 0.25 and own-UOV output gain >= 0.5 x the twin's (mean over
  heads); **BROKEN**: mean pure-pair correlation < 0.10 or own-UOV gain <= 0.25 x the twin's; otherwise PARTLY BROKEN.
- Cost **HIGH**: dev_full CE >= 2% above the twin; **LOW**: < 1%; otherwise MODERATE.

| | cost LOW | cost HIGH |
|---|---|---|
| law INTACT | GAMED: the penalised measurement is evaded; the aggregate law resists at no cost | PAID WITHOUT BREAKING: the model pays (for example score noise) but the law stays |
| law BROKEN | FREE BREAK: attention does its job without following its own vote's usefulness | COSTLY BREAK: the law is load-bearing |

Verdict for arm both: the cell met in at least 3 of 4 seeds; otherwise MIXED (each seed reported). Arms attn and vote,
if run, are classified the same way and reported as comparisons (no verdict weight).

**Restoring force** (per seed, after the continuation; twin continued the same way): recovery = (law strength of the
continued penalised model - of the penalised model) / (law strength of the continued twin - of the penalised model).
FAST if >= 0.5 in at least 3 of 4 seeds; NONE if <= 0.1 in at least 3 of 4; otherwise SLOW.

## Recorded expectations (no verdict weight)

DRAFT: the in-batch correlation is easy to suppress (it is noisy at instance level); the aggregate law weakens but does
not vanish, at a cost of 1-3%; a FAST return without the penalty. After development: PARTLY BROKEN with LOW cost in
most seeds (verdict MIXED) is the most likely outcome.

## Exploratory

The development grid's dose-response; arms attn and vote compared; where the function goes; checkpoints of the
continuation (how quickly the law returns); the in-batch correlation over training. No use of the test split.
