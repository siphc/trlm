# Study A, fixed-vote models: running log

Question, design and tests: logs/prereg_fixed_votes.md. Plan: logs/plan_attention_law_and_readability.md section 4.
Decision: logs/decisions.md item 10.

| phase | content | human gate |
|---|---|---|
| 0 | DRAFT pre-registration, model code, sanity tests, tables | G0 (design, DRAFT) and G1 (training code) |
| 1 | usefulness tables (model-free prediction, before any model); MPS throughput (R2, R3); LR check (R1) | |
| 2 | development: base and shift, seed 0; analysis code and pipeline checks; FINAL pre-registration | G2 |
| 3 | confirmatory: base and shift, seeds 1-4; tests; results | G3 |

## Phase 0 (2026-10-18)

- `src/train/fixed_votes.py`: FixedVoteLM (learned attention: embeddings, LayerNorm, W_Q, W_K; fixed tables as
  non-persistent buffers checked by sha; five scalars beta) and `build_tables` -> data/fixed_votes/tables.npz (sha
  135ef9807fb12aba; 96 MB; from the analysis cache of program.corpus.build on the TRAIN split).
- Hook in `src/train/train.py`: TrainConfig.fixed_tables ("" = TinyLM, unchanged; checked by the main-track test) and
  load_model recognises fixed-vote checkpoints. Commands in `src/train/runs.py`: `fixed_lr` (R1), `fixed --name base|shift`.
- `src/train/test_sanity_fixed_votes.py`: 9 tests pass on CPU (causality, tables are buffers without gradient and not
  in checkpoints, a numpy reference forward, identity attention gives self votes, sha check, bitwise resume and load,
  single-batch fit below the bigram, real tables match the corpus statistics and are normalised).
- One full-size step: 0.8 s on the CPU (4 threads, while the GPU trains the readable-training confirmatory models).

## Phase 1 (started 2026-10-18; G0/G1 not yet given)

- **M2 computed after the DRAFT commit, before any fixed-vote model** (`analysis/fixed_votes/usefulness.py`,
  analysis/out/fixed_votes/usefulness/, ~1.3 GB; log logs/runs/fixed_votes_usefulness.log). Mean usefulness per
  distance d = 1..7 (normalised tables): next 0.184, 0.040, 0.036, 0.026, 0.018, 0.008, 0.010; skip2 0.839, 0.261,
  0.152, 0.122, 0.089, 0.066, 0.053; skip3 0.277, 0.700, 0.301, 0.218, 0.169, 0.128, 0.107; topic rises 0.067 -> 0.243;
  copy -0.066 -> +0.020. Band 8-63: topic 0.161, copy 0.042, skip3 0.041, skip2 0.021, next 0.005.
  **Pre-registered predictions for H3a: head 3 peaks at d = 1 with skip2 (base) and at d = 2 with skip3 (shift).**
  N_1 recount differs from analysis/out/candidates/qk/N_1.npy by at most 1 count.
- `analysis/fixed_votes/law.py` (M1, M3-M6 on one model) and `analysis/fixed_votes/check_pipeline.py` (the DRAFT's two
  pipeline checks) written, **not yet run or tested**. A random-init checkpoint for the synthetic check:
  models/fixed_votes/scratch/init_s0/final.pt. **Both pipeline checks pass** (analysis/out/fixed_votes/
  pipeline_checks.json): goal 1's fit code reproduces attn_s1 head 0's stored base, own and tt KLs exactly (difference
  0.0); on the synthetic pattern the own arm recovers c1, c2, cb to within 1e-6 and 100% of the base rule's KL.
  law.py's full run on a trained model is still untested.
- DRAFT edits after the first commit (all before any model output): negative-beta rule for H3a; the synthetic check
  compares with the base rule's KL instead of the token-token ceiling.

## Next steps (for whoever resumes)

1. Ask the human for G0/G1 on this study (DRAFT + code + tests). Then, when the GPU is free (the readable-training
   confirmatory queue runs first): MPS step time (R2, R3), `python -m src.train.runs fixed_lr --device mps` (R1), record
   R1 in analysis/out/fixed_votes/rules.json, then `fixed --name base --seeds 0` and `--name shift --seeds 0` with the
   chosen LR (development), run law.py on them, the pipeline checks, then the FINAL pre-registration and gate G2.

## 2026-10-19 (G0/G1 approved, decision 12)

- DRAFT edit before any output: confirmatory runs all on one hardware (the cloud instance if available); development
  (R1, seed 0) may run on the laptop GPU. `analysis/fixed_votes/rules.py` (R1, R2). Development runs are queued in
  `scripts/next_studies/dev_queue.sh` after study B's (R2 timing on MPS, R1 LR check, base and shift seed 0).
- Cloud kit in scripts/aws/ (README there); `scripts/aws/make_jobs.py A` writes the confirmatory job files once the
  FINAL is committed.
- 2026-10-20 00:35: `law.py` smoke test on the random-initialised checkpoint ran end to end (all arms, pure-pair
  correlations, vote removal; values meaningless at initialisation: near-uniform attention, all gains 0). About 35 min
  per model on the CPU while the GPU queues run.
- **R1 (2026-10-20 03:00): peak LR = 4e-3** (dev_full CE after 50M tokens, base tables, seed 0, laptop GPU: 1e-3
  2.6120, 2e-3 2.5874, 4e-3 2.5775). The best value is the largest of the pre-registered grid, so the optimum may lie
  higher; applied as written. For scale: attn_s0..s4 reach 2.491-2.540 after a full epoch. Laptop step time 0.21 s
  (R2_mps; R2 binds on the instance). Development runs base_s0 and shift_s0 (one epoch, LR 4e-3) started 03:00.
- **Development: base, seed 0** (one epoch, LR 4e-3, laptop GPU, 3.98 h; law.py on the CPU; no verdict weight):
  dev_full CE 2.5428 (attn_s0 2.5095: +1.3%); beta_follow 0.93; head vote strengths next 0.70, copy 0.68, topic 1.51,
  skip2 1.78; vote-removal shifts 0.060, 0.081, 0.249, 0.481 (all active).

  | head | own-usefulness gain | ceiling tt (tt_self) | own / tt | largest other / own | pure-pair: own (largest other) | attention peak d=1..7 (mass at peak) | coef c1, c2, cb |
  |---|---|---|---|---|---|---|---|
  | next | 0.0090 | 0.0368 (0.0472) | 0.24 | 0.05 | 0.361 (0.021) | 1 (0.20) | 2.22, 2.30, 3.24 |
  | copy | 0.0130 | 0.0394 (0.0394) | 0.33 | 0.49 (topic) | -0.035 (0.054) | flat, ~0.01 | 0.30, 0.41, 1.25 |
  | topic | 0.0106 | 0.0909 (0.0894) | 0.12 | 0.62 (copy) | 0.203 (0.131) | 7, rising | 0.29, 0.48, 0.45 |
  | skip2 | 0.0367 | 0.0849 (0.0894) | 0.43 | 0.09 | 0.433 (0.054; skip3 alt 0.112) | **1 (0.53)** | 0.43, 0.59, 0.49 |

  Reading: head-specific and as strong as in normal models by the pure-pair correlation for the locally looking heads
  (next 0.36, skip2 0.43; normal 0.34-0.44); but own usefulness recovers only 12-43% of the token-token ceiling's output
  gain (P1' would fail for every head at seed 0; normal models: >= 50% in 7 of 8 local heads). copy and topic partly
  follow each other's tables. Each head's attention-by-distance matches its table's usefulness profile (skip2 peaks at
  d = 1 as predicted for H3a). The fixed-vote model does not make the law a larger share of attention.
