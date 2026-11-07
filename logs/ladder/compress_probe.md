# Exploratory: how many numbers does L1's behaviour at the seed floor need? (2026-11-06)

Follow-up to decision 18. analysis/ladder/compress_probe.py fits structured versions of L1 (ln arm, seed 0, 644
parameters) to the model's outputs (KL objective, 2,000 steps of 64 train windows) and scores them on dev_full.
Seed floor (KL to the average of seeds 1-4): **0.0901**. Not an account: no claims, no stated meaning.

| version | free numbers | KL | at or below the floor |
|---|---|---|---|
| model refit (sanity) | 644 | 0.0000 | yes |
| positions 2 (position 0, one vector for the rest) | 396 | 0.0146 | yes |
| groups 4 / 8 / 16 / 32, no offset | 344 / 360 / 392 / 456 | 0.940 / 0.701 / 0.354 / 0.124 | no |
| groups 4 / 8 / 16 / 32 + per-symbol output offset | 423 / 439 / 471 / 535 | 0.417 / 0.270 / 0.129 / 0.036 | only 32 |
| groups 16 + offset, positions 2 | 223 | 0.130 | no |
| **groups 32 + offset, positions 2** | **287** | **0.049** | **yes** |

Groups: frequency-weighted k-means on the model's embedding; the symbol-to-group assignment (79 symbols) is not
counted in the free numbers (about 6 bits per symbol for 32 groups).

Reading:
- The 64 position vectors are nearly redundant: two vectors cost 0.015.
- Merging symbols costs the most; a per-symbol output offset (a unigram-like correction) halves the cost at every k.
- About 290 numbers plus a 79-entry group table reach the floor: roughly half the model's numbers, a compression of
  about 2 (against the main track's distillation, which needed more numbers than its model).

## Greedy search, L1-L3, both arms, seed 0 (2026-11-06)

`compress_probe.py` main() now searches greedily, scoring on dev_full[:1000] with 2,000 fitting steps per version.
The steps are tried in a fixed order:
1. positions 2;
2. halve the per-head QK/OV rank (`rank r`: each head's QK and OV are truncated to rank r by SVD, then fitted),
   starting from d_head;
3. groups 64, 48, 32, 24, 16, 12, 8, each with the per-symbol offset.

A step is kept only if the version's KL stays at or below the bar, max(floor, 0.01). Each step builds on the steps
kept before it. The full per-version rows are in analysis/out/ladder/compress_search_<rung>_<arm>_s0.json; the log
is logs/runs/ladder_compress_L1_L3.log.

| rung | arm | params | floor (bar) | smallest version at the bar | free numbers | KL | params / free |
|---|---|---|---|---|---|---|---|
| L1 | ln | 644 | 0.090 | positions 2, rank 1, groups 32 + offset | 239 | 0.080 | 2.69 |
| L1 | noln | 636 | 0.229 | rank 1, groups 32 + offset | 479 | 0.198 | 1.33 |
| L2 | ln | 1416 | 0.097 | rank 2, groups 48 + offset | 1055 | 0.097 | 1.34 |
| L2 | noln | 1400 | 0.069 | rank 2, groups 64 + offset | 1167 | 0.065 | 1.20 |
| L3 | ln | 1416 | 0.111 | rank 2, groups 48 + offset | 1119 | 0.061 | 1.27 |
| L3 | noln | 1400 | 0.120 | rank 2, groups 32 + offset | 975 | 0.091 | 1.44 |

Single steps, from each rung's search:

| rung | arm | positions 2 | rank d_head/2 | rank d_head/4 |
|---|---|---|---|---|
| L1 | ln | 0.014 | 0.017 (r 2) | 0.045 (r 1) |
| L1 | noln | 0.234 (fails) | 0.005 (r 2) | 0.203 (r 1) |
| L2 | ln | 0.175 (fails) | 0.005 (r 4) | 0.086 (r 2) |
| L2 | noln | 0.175 (fails) | 0.001 (r 4) | 0.067 (r 2) |
| L3 | ln | 0.185 (fails) | 0.044 (r 2) | 0.163 (r 1, fails) |
| L3 | noln | 0.250 (fails) | 0.036 (r 2) | 0.181 (r 1, fails) |

At L1 the rank columns build on positions 2.

Reading:
- **Compression shrinks after L1.** L1 ln compresses about 2.7 times; L2 and L3 compress about 1.2-1.4 times.
- **The floors are wide at these rungs** (0.07-0.23 nats), so the bar has slack: most of the compression is the
  slack being used up, not structure.
- **At a tight bar, almost nothing compresses.** Only one simplification costs under 0.01 at L2: halving the rank of
  a single-head model (8 to 4), which costs 0.001-0.005. Every other simplification at L2-L3 costs 0.04 or more.
- **Positions matter from L2 on.** The two-vector position code works only at L1 ln. At L2 and L3 it costs 0.17-0.25.
- **The count is generous to the versions.** Free numbers are floats, not bits. The low-rank factors carry gauge
  freedom (about r² numbers per factor pair), so the true count is a little lower than shown. The 79-entry group
  assignment is not counted.
- **L4-L7 (ln, seed 0) are running**, with logs in logs/runs/ladder_compress_L4_L5.log and _L6_L7.log.
