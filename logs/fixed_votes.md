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
