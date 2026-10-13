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

## Gradual pruning with 50% of the budget reserved for FOLLOW (tag gp60k_f50)

| round | budget | kept FOLLOW | kept RECALL (h0, h1, h2, h3) | KL after cut | KL after refit |
|---|---|---|---|---|---|
| 0 | 9,675,427 | 3,520,000 | 964,079 / 1,586,802 / 1,052,323 / 1,234,509 | 0.2349 | 0.2212 |
| 1 | 4,680,695 | 2,340,348 | 395,720 / 865,867 / 492,897 / 585,863 | 0.2561 | 0.2403 |
| 2 | 2,264,386 | 1,132,193 | 162,239 / 467,595 / 231,034 / 271,325 | 0.3177 | 0.2861 |
| 3 | 1,095,445 | 547,722 | 67,328 / 252,017 / 105,652 / 122,726 | 0.4256 | 0.3702 |
| 4 | 529,945 | 264,972 | 27,430 / 135,145 / 47,500 / 54,898 | 0.5957 | 0.5019 |
| 5 | 256,372 | 128,186 | 11,632 / 70,371 / 21,180 / 25,003 | 0.8077 | 0.6759 |
| 6 | 124,026 | 62,013 | 5,407 / 35,008 / 9,664 / 11,934 | 0.9359 | 0.7987 |
| 7 | 60,000 | 30,000 | 2,653 / 16,735 / 4,431 / 6,181 | 1.0026 | 0.8435 |

Reserving FOLLOW's share cuts KL at 60k from 1.259 to 0.844 (dev_fast harness: 0.8499, iteration v6) and at
124k from 1.117 to 0.799. Allocation across components matters as much as the pruning schedule.

## Gradual pruning, Fisher ranking, heads silent on the current token (tag gp60k_fisher_nr) — v7

Idea from the design panel's mechanism proposal: (1) heads cast no vote through the current token
(`make_nonredundant` moves each head's average current-token vote into FOLLOW; dense KL 0.2233 -> 0.2496), so
every current-token fact has one home; (2) rank exceptions by the second-order removal cost
0.5 w^2 sum a^2 q(1-q) instead of the first-order |w| sum a q, which overvalued the diffuse head 1 by ~1/a.
Importance from 100 batches of 16 windows; 8 rounds; 300 refit steps (1000 last); attention trainable.

| round | budget | kept FOLLOW | kept RECALL (h0, h1, h2, h3) | KL after cut | KL after refit |
|---|---|---|---|---|---|
| 0 | 9,675,427 | 2,503,350 | 1,581,764 / 1,937,373 / 1,782,737 / 1,870,203 | 0.2525 | 0.2261 |
| 1 | 4,680,695 | 1,343,506 | 650,697 / 909,236 / 871,899 / 905,357 | 0.2609 | 0.2357 |
| 2 | 2,264,386 | 635,927 | 290,834 / 432,949 / 456,940 / 447,736 | 0.3095 | 0.2646 |
| 3 | 1,095,445 | 302,966 | 135,536 / 196,306 / 239,342 / 221,295 | 0.3472 | 0.3042 |
| 4 | 529,945 | 158,140 | 64,731 / 79,570 / 121,756 / 105,748 | 0.3894 | 0.3494 |
| 5 | 256,372 | 86,756 | 30,791 / 30,383 / 59,024 / 49,418 | 0.4374 | 0.4006 |
| 6 | 124,026 | 48,518 | 14,475 / 11,454 / 27,305 / 22,274 | 0.4967 | 0.4596 |
| 7 | 60,000 | 27,224 | 6,691 / 4,225 / 12,181 / 9,679 | 0.5711 | 0.5138 |

At 60k: 0.514 held-out, dev_fast harness 0.5165 (v6 at the same budget: 0.8499). The ranking rule was the
dominant problem: at every budget below 1M the Fisher curve is far below the linear one.

## Same, smoother schedule: 12 rounds, 400 refit steps, 2000-step final refit (tag gp60k_fisher_nr_r12) — v8

| round | budget | kept FOLLOW | kept RECALL (h0, h1, h2, h3) | KL after cut | KL after refit |
|---|---|---|---|---|---|
| 0 | 12,325,090 | 2,912,685 | 2,207,739 / 2,471,168 / 2,328,540 / 2,404,958 | 0.2508 | 0.2222 |
| 1 | 7,595,392 | 2,043,548 | 1,182,282 / 1,484,593 / 1,395,985 / 1,488,984 | 0.2286 | 0.2194 |
| 2 | 4,680,695 | 1,291,940 | 672,568 / 910,232 / 877,724 / 928,231 | 0.2427 | 0.2267 |
| 3 | 2,884,499 | 750,769 | 396,118 / 565,611 / 579,214 / 592,788 | 0.2686 | 0.2422 |
| 4 | 1,777,586 | 442,244 | 232,704 / 346,920 / 381,273 / 374,445 | 0.2899 | 0.2621 |
| 5 | 1,095,445 | 279,210 | 140,491 / 198,589 / 246,346 / 230,809 | 0.3111 | 0.2847 |
| 6 | 675,073 | 185,053 | 85,071 / 108,963 / 155,187 / 140,799 | 0.3354 | 0.3105 |
| 7 | 416,017 | 126,208 | 51,827 / 58,196 / 95,232 / 84,554 | 0.3615 | 0.3385 |
| 8 | 256,372 | 86,780 | 31,485 / 29,400 / 58,088 / 50,619 | 0.3932 | 0.3717 |
| 9 | 157,991 | 59,309 | 18,940 / 15,225 / 34,692 / 29,825 | 0.4307 | 0.4084 |
| 10 | 97,362 | 40,267 | 11,299 / 7,927 / 20,628 / 17,241 | 0.4734 | 0.4516 |
| 11 | 60,000 | 27,571 | 6,645 / 4,003 / 12,023 / 9,758 | 0.5274 | 0.4885 |

60k: 0.4885 held-out, dev_fast harness 0.4919 (v7: 0.5165). Smoother schedules help at every budget.
