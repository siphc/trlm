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

## 2026-10-13

5. **Verdict on the distillation program (v11):** the model-fitted tables read like reverse-engineered weights and
   give little research insight; only the scaffolding is readable. The fitted-table approach ends here; the state is
   tagged `distillation-v11` (commit 96c60a9) as a return point.
6. **New direction, "explain, don't fit"** (agent's plan, accepted): token-level knowledge only from transparent
   sources (statistics counted from the training corpus by stated rules, human-named word classes); numbers fitted to
   the model only as a few named scalars per mechanism (working cap: a few hundred in total); every mechanism starts
   as a hypothesis from circuit analysis and is kept only if it reduces KL and matches the model's internals; the
   headline result is a mechanism ledger (what each mechanism explains) plus a characterised residual.
7. **One-time exception to AGENTS.md section 8 "no file I/O at runtime":** the program may use the TRAIN-split corpus
   at startup to compute its statistics. Implementation: the harness reads `data/tokens/train.bin` and passes it to
   `program.setup(corpus)` once before scoring (the program itself still does no file I/O and the lint is unchanged);
   the sandbox allows reading that one file only. Dev/test text and models stay unreadable. configs/frozen.yaml v3.
