# Coverage of the program against the seed floor (decision 14)

Measured 2026-11-06 with analysis/program_gap/coverage.py on fitlib.train_windows(64, seed=2000) (the windows of
ensemble_target.py and seed_floor.py; TRAIN split, no dev or test). Output: analysis/out/program_gap/coverage.json.
Program: program/ at v1.3 (fitted to attn_s0 only).

## Definition

For each seed k, KL(attn_s{k} || x) per token, with:
- **floor**: the probability average of the other four seeds. It is what a description that does not know seed k's own
  training run could at best be expected to match (a single sibling seed is a weaker reference, reported too);
- **baselines**: the harness's bigram and trigram baselines (baselines/);
- **coverage** = (KL_bigram - KL_program) / (KL_bigram - KL_floor): the share of the way from the bigram to the floor.

## Numbers

| seed | bigram | trigram | program v1.3 | floor (other 4) | single siblings (mean) | coverage vs bigram | vs trigram |
|---|---|---|---|---|---|---|---|
| attn_s0 | 1.342 | 1.208 | 0.547 | 0.248 | 0.404 | 0.727 | 0.688 |
| attn_s1 | 1.308 | 1.186 | 0.544 | 0.265 | 0.415 | 0.732 | 0.697 |
| attn_s2 | 1.337 | 1.252 | 0.560 | 0.282 | 0.436 | 0.737 | 0.714 |
| attn_s3 | 1.318 | 1.170 | 0.540 | 0.238 | 0.388 | 0.720 | 0.676 |
| attn_s4 | 1.322 | 1.196 | 0.543 | 0.245 | 0.398 | 0.723 | 0.687 |
| mean | 1.325 | 1.202 | 0.547 | 0.256 | | **0.728** | 0.692 |

From the consensus side: KL(five-seed average || program) 0.411 +- 0.005; KL(average of the other four || seed k)
0.283 on average. Against a single sibling as the reference, coverage would read 0.849.

## How to read it

- The program covers about 73% of the way from the bigram to what the other seeds agree on, the same for every seed
  although it was fitted to attn_s0: it describes shared behaviour, nothing seed-specific.
- The remaining 0.29 nats (0.547 - 0.256) are behaviour the other seeds share with seed k and the program lacks. The
  largest known part is copying: 58% of the program's miss against the five-seed consensus is at positions whose
  consensus top token appeared earlier in the window (analysis/out/program_gap/consensus.json; logs/mech_v3_report.md
  1c). That is the next target (plan: copying diagnostic, then candidates).
- Not a harness metric: the harness's primary metric stays KL(model || program) on dev.

## Where the remaining gap is: oracle swaps (analysis/program_gap/copy_diagnostic.py, 2026-11-06)

Swaps of attn_s0's exact parts into the program (all strengths refitted to attn_s0 as in
analysis/token_geometry/tables_vs_attention.py; fit train_windows(16, 2024), evaluation train_windows(32, 2000)),
scored against attn_s0 and against the five-seed consensus ENS5. Change in KL against the refitted program A (A: 0.551 to
attn_s0, 0.412 to ENS5); paired SE over 32 windows. Outputs: analysis/out/program_gap/copy_diagnostic.json and
copy_diagnostic_all_heads.json.

| head made exact inside the program | attention only (to s0 / to ENS5) | vote table only | both |
|---|---|---|---|
| head 0 (LOCAL-0) | -0.001 / +0.001 | +0.001 / +0.004 | -0.025 / -0.008 |
| head 1 (RECALL) | -0.027 / -0.023 | +0.006 / +0.020 | -0.071 / -0.035 |
| head 2 (LOCAL-2) | -0.001 / +0.001 | +0.002 / +0.007 | -0.032 / -0.006 |
| head 3 | -0.001 / -0.000 | -0.003 / -0.001 | -0.035 / -0.010 |
| all parts exact (D) | | | -0.436 / -0.129 |

SEs 0.001-0.005 (D: 0.012 / 0.007). The direct-path table in place of FOLLOW alone makes the program far worse (+0.52).

- **Copying is not the main gap.** The class "the consensus top token appeared earlier in the window" (59% of
  positions in this sample, 59% of the miss to ENS5) is mostly common tokens, not copying: every seed's RECALL head
  puts only 3-7% of its attention on earlier occurrences of that token there (the program 4-6%; seeds agree with each
  other at r = 0.60 over positions, the program with them at 0.37-0.47). Making head 1 fully exact closes 0.035 of the
  0.412 miss to the consensus.
- **No single part carries the gap.** Each head made exact, attention and table together, closes 0.006-0.035 nats
  against the consensus; their sum is about 0.06, while all parts exact close 0.129. Exact attention alone helps only
  for head 1, and an exact table alone makes things worse for every head (the lock, again at the level of single
  parts). What the seeds share and the program lacks is a property of how the parts work together, not of a part that
  could be described on its own.
- Consequence for the plan: no copying candidate has room above the adoption bar (the oracle ceiling against the
  consensus is 0.035 for RECALL and at most 0.010 for any other head). The copying line stops here (no candidates
  built).
