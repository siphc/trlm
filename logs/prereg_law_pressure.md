# Pre-registration (DRAFT): can the look-follows-write law be Goodharted? (study B, `law_pressure`)

DRAFT, 2026-10-18. Committed before any model is trained with the penalty. Decision: logs/decisions.md item 10. Plan:
logs/plan_attention_law_and_readability.md section 5. Code: src/train/law_pressure.py (penalty), src/train/train.py
(hook, default off), src/train/runs.py (`law_dev`, `law`, `law_restore`), src/train/test_sanity_law_pressure.py.
Running log: logs/law_pressure.md.

## What has been seen before writing this

- The law in normal models (logs/look_follows_write_results.md and addendum): per local head, the pair-specific part of
  attention follows the usefulness of the head's own vote, aggregated over the corpus (pure-pair correlation 0.34-0.44,
  other heads' at most 0.08; own usefulness gains 0.06-0.18 nats in the output test); about 0.24 attention score per
  logit of usefulness.
- The penalty's trained quantity in the normal models: the pooled in-batch correlation of attention scores and
  instance usefulness (definition below) on train_windows(16, 9300) is positive for every head of attn_s0..s4:
  0.106-0.161 (attn_s0: 0.131, 0.143, 0.132, 0.110).
- Sanity tests (10 pass on CPU): the terms equal a direct computation from the attention scores and OV tables; the arms
  route gradients as intended; lambda = 0 is bitwise the plain run; resume is bitwise; optimising the penalty alone on
  one batch of attn_s1 removes most of the correlation (squared correlation below a quarter of the start in 60 steps).
- Nothing about models trained with the penalty.

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

## Decision rules (development only; values go into the FINAL)

- R1: lambda* = the smallest grid value whose 100M-token run (arm both) has a mean over heads of |in-batch correlation|
  (on train_windows(16, 9300), src/train/law_pressure.corr_on) at most 20% of the control's; the largest grid value if
  none does. No cost cap: the cost is an outcome.
- R2: if a penalised step takes more than 1.6x the plain step on the study's hardware, the penalty uses the first 16
  windows of each batch (TrainConfig.law_windows).
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

The in-batch correlation is easy to suppress (it is noisy at instance level). The aggregate law weakens but does not
vanish, at a cost of 1-3%; a FAST return without the penalty.

## Exploratory

The development grid's dose-response; arms attn and vote compared; where the function goes; checkpoints of the
continuation (how quickly the law returns); the in-batch correlation over training. No use of the test split.
