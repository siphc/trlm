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
