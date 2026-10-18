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
