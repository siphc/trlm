# analysis/ladder: model access for the ladder of tiny language models

The models: `src/train/ladder_runs.py` (rungs, pilot, chosen configs), data in `src/train/data_char.py`, results in
`logs/ladder/` (pilot.md, trained.md, dumps/).

- `common.py`: load a rung's model, eval windows, KL; `parts`/`finish` split a one-block attention-only model into
  its direct path x0 and its heads' outputs z and put them back together; `resampled` resample-ablates one site.
- `eval.py`: dev loss, KL to the unigram, bigram and trigram baselines, the seed floor; `--check_l10` reproduces the
  harness's KL for program v1.3.
- `dump.py`: a readable dump of every weight table and the site importances.
- `reference_forward.py`: a NumPy forward pass from the weights (the base of the interval version in
  `analysis/proofs/`).
- `test_tools.py`: self-tests on a pilot model.

The account tools of the closed account study (logs/ladder_accounts/) are at git tag pre-proofs.
