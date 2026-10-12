# Compression experiments (attn_s0)

Held-out KL = mean per-token KL(model || student) on 200 held-out TRAIN windows (seed 2000), unless noted.
Reference: dense student (all 20M exceptions free) 0.2233; class-level only 1.666.

## Gradual pruning, global budget, data-importance ranking (`analysis/prune.py`, tag gp60k)

8 rounds, geometric budgets, 300 refit steps per round (1000 in the last), lr 0.006, from
`student_dense_ref_dense.pt`.

| round | budget | kept FOLLOW | kept RECALL (h0, h1, h2, h3) | KL after cut | KL after refit |
|---|---|---|---|---|---|
| 0 | 9,675,427 | 1,279,493 | 1,915,904 / 2,413,023 / 1,927,181 / 2,139,826 | 0.2277 | 0.2169 |
| 1 | 4,680,695 | 453,619 | 814,852 / 1,420,016 / 913,537 / 1,078,671 | 0.2522 | 0.2341 |
| 2 | 2,264,386 | 155,756 | 346,286 / 793,134 / 443,827 / 525,383 | 0.3090 | 0.2798 |
| 3 | 1,095,445 | 54,091 | 146,353 / 437,150 / 211,716 / 246,135 | 0.4039 | 0.3613 |
| 4 | 529,945 | 18,673 | 61,409 / 239,423 / 98,543 / 111,897 | 0.5664 | 0.5067 |
| 5 | 256,372 | 5,705 | 25,862 / 128,463 / 45,358 / 50,984 | 0.8445 | 0.7572 |
| 6 | 124,026 | 1,069 | 11,714 / 66,343 / 20,960 / 23,940 | 1.2339 | 1.1174 |
| 7 | 60,000 | 157 | 5,879 / 32,372 / 9,764 / 11,828 | 1.4326 | 1.2593 |

Findings: gradual pruning holds fidelity down to ~1M entries (0.36) but at 60k (1.26) it is no better than
v4 (1.16 held-out with 20k FOLLOW exceptions and class-level heads). The data-importance ranking starves
FOLLOW (157 left) in favour of RECALL (head 1 keeps the most); masks are monotone, so information cannot move
back into FOLLOW once its entries are cut. Allocation needs a protected FOLLOW share, and the low-budget
regime needs a different representation.
