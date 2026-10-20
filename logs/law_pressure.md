# Study B, training against the look-follows-write law: running log

Question, design and tests: logs/prereg_law_pressure.md. Plan: logs/plan_attention_law_and_readability.md section 5.
Decision: logs/decisions.md item 10.

| phase | content | human gate |
|---|---|---|
| 0 | DRAFT pre-registration, penalty code, sanity tests | G0 (design, DRAFT) and G1 (training code) |
| 1 | MPS step time (R2); development grid (seed 0, 100M tokens) and R1, R3; law tools for checkpoint paths; FINAL | G2 |
| 2 | confirmatory runs (seeds 1-4), continuations, measurements, results | G3 |

## Phase 0 (2026-10-18)

- `src/train/law_pressure.py`: the penalty (instance usefulness computed as A_h(t_j) . (E(y_i) - (P1 E)(t_i)), so
  nothing V x V is formed per pair; gathers through F.embedding because advanced indexing's backward was
  nondeterministic on CPU at 1e-7, found by the resume test), `corr_on` for offline measurement.
- Hooks in `src/train/train.py` (law_arm, lambda_law, law_ramp, law_p1_path, init_from; all default off; the
  main-track test passes) and commands in `src/train/runs.py` (`law_dev`, `law`, `law_restore`).
- `src/train/test_sanity_law_pressure.py`: 10 tests pass on CPU.
- Baseline in-batch correlation on train_windows(16, 9300): attn_s0 0.131/0.143/0.132/0.110; attn_s1..s4 0.106-0.161,
  every head positive.
- Law tools accept checkpoint paths (analysis/look_write/common.out_name, commit 507144a).

## Next steps (for whoever resumes)

1. Ask the human for G0/G1 (DRAFT + code + tests). Then on MPS: step time with the penalty vs plain (R2); the
   development grid `python -m src.train.runs law_dev --device mps` (arm both, lambda 1..100, 100M tokens); measure
   corr_on (train_windows(16, 9300)) on each and on models/readable/dev/none_s0_100M; apply R1, then arms attn and vote
   at lambda* (`law_dev --arms attn,vote --lams <lambda*>`), R3; uov_tables + fit_arms on the dev models (paths); FINAL.

## Phase 1 (2026-10-19; G0/G1 approved, decision 12)

- DRAFT edits before any output: twins are lambda-0 runs on the same hardware; development grid includes lambda 0 as
  its control; each comparison within one hardware (development on the laptop GPU, confirmatory runs on the cloud
  instance if available); R2's option `law_windows`.
- `analysis/law_pressure/rules.py` (R1-R3) and `scripts/next_studies/dev_queue.sh` (B's development, then A's).
- **R2 (MPS, 19:04):** plain 0.096 s/step, penalised 0.247 s/step (ratio 2.56 > 1.6), measured while the
  readable-training run shared the GPU (both timings equally affected) -> **law_windows = 16**. Because lambda* is chosen
  with this penalty, every run of study B (development, confirmatory, continuations) uses 16 windows, on any hardware.
- Development grid started 19:05 (lambda 0, 1, 3, 10, 30, 100; seed 0; 100M tokens; log logs/runs/next_studies_dev.out).
- Cloud kit in scripts/aws/ (README there); `scripts/aws/make_jobs.py B` writes the confirmatory job files (twins,
  penalised runs, continuations, law tools) once the FINAL is committed.
- 2026-10-19 19:46: development queue paused at the human's request. Grid run lambda 0 finished; lambda 1 resumes from
  latest.pt (19:45, about step 5,000 of 6,103). Resume: `nohup caffeinate -dimsu scripts/next_studies/dev_queue.sh mps
  >> logs/runs/next_studies_dev.out 2>&1 &` and `nohup scripts/next_studies/watchdog.sh scripts/next_studies/dev_queue.sh
  > /dev/null 2>&1 &` (finished steps are skipped; R2 is not re-measured).
- Development: goal 1's law tools on the lambda = 0 control (models/law_pressure/dev/both_l0_s0_100M, 100M tokens;
  10 min on the CPU; first use with a checkpoint path): heads LOCAL, RECALL, LOCAL, RECALL; pure-pair correlation with
  own usefulness +0.398, +0.252, +0.368, +0.261 (mean 0.320; full-epoch normal models 0.326-0.351), other heads and
  shuffled controls within +-0.08; own-usefulness gains +0.122, +0.025, +0.046, +0.036 (mean 0.057; full-epoch
  0.070-0.090). The law is present at 100M tokens, so the development grid can show whether the penalty removes it.
- **R1 (2026-10-20 01:14): lambda* = 1** (the smallest grid value meeting the 20% criterion). Development grid (seed 0,
  100M tokens, penalty on 16 windows; in-batch correlation on train_windows(16, 9300); goal 1's law tools):

  | lambda | mean abs in-batch corr (share of control) | pure-pair corr, own (mean of 4 heads) | own-usefulness gain (mean) | dev_full CE (cost) |
  |---|---|---|---|---|
  | 0 (control) | 0.110 (1.00) | 0.320 (0.398, 0.252, 0.368, 0.261) | 0.057 | 2.6618 |
  | 1 | 0.018 (0.16) | 0.175 (0.308, 0.090, 0.272, 0.028) | 0.048 | 2.6676 (+0.22%) |
  | 3 | 0.011 (0.10) | 0.189 | 0.051 | 2.6526 (-0.35%) |
  | 10 | 0.006 (0.05) | 0.142 | 0.048 | 2.6462 (-0.59%) |
  | 30 | 0.005 (0.05) | 0.140 | 0.044 | 2.6640 (+0.08%) |
  | 100 | 0.004 (0.035) | (law tools running) | | 2.6833 (+0.81%) |

  Reading (development only, one seed, 100M tokens): the penalised instance statistic falls 6-30x at no measurable
  cost; the aggregate law (never seen by the penalty) roughly halves and then plateaus near 0.14; the own-usefulness
  output gain stays at 77-89% of the control's. Head-specificity weakens at lambda 1-3 (another head's usefulness
  reaches |0.23| pure-pair correlation; control <= 0.08). On the pre-registered definitions seed 0 would be PARTLY
  BROKEN with LOW cost. The outcome thresholds stay as in the DRAFT (fixed before any output).
- **R3 (01:50): neither arm admitted** (at lambda 1 arm attn reaches 51% of the control's in-batch correlation, arm vote
  72%; only the joint penalty meets the 20% criterion). Development law tools: lambda 100 pure-pair 0.256, own gain
  0.015; arm attn pure-pair 0.179, own gain 0.050; arm vote pending (exploratory).
- **FINAL pre-registration written (2026-10-20); waiting for gate G2.** Confirmatory: twins and lambda 1, seeds 1-4,
  continuations, law tools (job files: `scripts/aws/make_jobs.py B`).
