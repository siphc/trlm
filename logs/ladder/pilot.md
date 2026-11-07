# Ladder pilot (logs/ladder_accounts/design.md, Training)

Seed 0, arm ln. LR check at the smallest budget; budget ladder at the best LR, each run with its own full cosine schedule. Dev loss is the mean cross-entropy on dev_full (nats per symbol).

## L1

| peak LR | budget (symbols) | dev_full loss | steps | seconds |
|---|---|---|---|---|
| 0.001 | 25,000,000 | 2.6856 | 6,103 | 22 |
| 0.003 | 25,000,000 | 2.6283 | 6,103 | 21 |
| 0.01 | 25,000,000 | 2.5924 | 6,103 | 21 |
| 0.03 | 25,000,000 | 2.5497 | 6,103 | 11 |
| 0.09 | 25,000,000 | 2.6157 | 6,103 | 11 |
| 0.03 | 100,000,000 | 2.5500 | 24,414 | 50 |
| 0.03 | 400,000,000 | 2.5268 | 97,656 | 211 |

**Chosen for L1:** peak LR 0.03, budget 25,000,000 symbols (smallest within 1% of the best, 2.5268).

## L5

| peak LR | budget (symbols) | dev_full loss | steps | seconds |
|---|---|---|---|---|
| 0.001 | 25,000,000 | 1.8606 | 6,103 | 30 |
| 0.003 | 25,000,000 | 1.7424 | 6,103 | 28 |
| 0.01 | 25,000,000 | 1.6670 | 6,103 | 41 |
| 0.03 | 25,000,000 | 1.6672 | 6,103 | 43 |
| 0.01 | 100,000,000 | 1.6413 | 24,414 | 217 |
| 0.01 | 400,000,000 | 1.6175 | 97,656 | 791 |
| 0.01 | 1,884,119,040 | 1.6106 | 459,990 | 2906 |

**Chosen for L5:** peak LR 0.01, budget 400,000,000 symbols (smallest within 1% of the best, 1.6106).
