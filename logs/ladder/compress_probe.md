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
