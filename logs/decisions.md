# Human decisions

## 2026-10-12 (evening)

1. **Table size:** optimise and present the main line at ~60k exceptions (v9: ~106k numeric literals).
   Larger and smaller budgets stay as curve points only.
2. **Harness:** approved to extend harness/claims.py (and its helper in harness/common.py) with MEAN ablation
   and earlier-token-only ablation of a model head, for claim tests. The scorer (eval_fidelity) is unchanged.
3. **Protocol:** fidelity changes are adopted on paired held-out TRAIN comparisons at a fixed budget
   (analysis/paired.py; gain >= 0.005 and >= 3 SE, >= 0.01 for a new class system or component). Only those
   fixed-budget fidelity iterations count toward the stopping rule; curve points, readability and explore commits
   do not. dev_fast is still run and logged for every commit.
4. **Concepts (section 8):** analysis only. Head-1 concepts may be used to write claims and to group head-1
   exceptions under named comments, not as a runtime mechanism.

Earlier: frozen.yaml v2 (training on the laptop's MPS GPU), approved 2026-10-12.
