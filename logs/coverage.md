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
