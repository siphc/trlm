# analysis/ladder: the ladder of tiny language models

Question: what is the largest model of this family that can be understood completely, and what breaks first one step
past it? Design and criteria: `logs/ladder_accounts/design.md` (FINAL, decision 16).

- `eval.py`: P1 (KL against the seed floor), coverage, baselines; `--check_l10` reproduces the harness.
- `scrub.py`: P3, causal scrubbing of an account's `CLAIMS`.
- `length.py`: R1, description length of an account and of the quantised weights; the account rules.
- `reference_forward.py`: the NumPy forward pass counted in the weights reference.
- `common.py`: models, accounts, KL.
- `test_tools.py`: self-tests on a pilot model.

Training: `src/train/ladder_runs.py` (rungs, pilot); data: `src/train/data_char.py`. Results: `logs/ladder/`.
