# look_write: pre-registered test of "where a head looks comes from what it writes"

Pre-registration: `logs/prereg_look_follows_write.md`. Results: `RESULTS.md` (narrative, definitions, verdict),
`results.json` (all numbers), `tables.md` (generated tables). Written independently of
`analysis/investigations/qk/` (only the seed-independent pair counts `analysis/out/candidates/qk/N_d.npy` are
read, as a check against our own recount).

Tables are cached under `analysis/out/look_write/<model>/` (that folder carries its own `.gitignore`).
Every step is one process at a time, `OMP_NUM_THREADS=2`, corpus in chunks of 10M positions.

## Run order

```bash
export OMP_NUM_THREADS=2
# 0. checks: substitution forward pass reproduces the model; numpy score formula; N_1 recount on 25M tokens
.venv/bin/python -m analysis.look_write.selftest
# 1. corpus pass: UOV_h,d and token-shuffled UOV for d = 1..7, all heads (~3 min per model, ~0.9 GB per model)
.venv/bin/python -m analysis.look_write.uov_tables attn_s1 attn_s2 attn_s3 attn_s4 attn_s0
# 2. LOCAL/RECALL, base rule, arms, exact-model substitution, pure-pair correlations (writes <model>/arms.json)
.venv/bin/python -m analysis.look_write.fit_arms attn_s1 attn_s2 attn_s3 attn_s4 attn_s0
# 2b. exploratory ceiling variants (sensitivity of P1)
.venv/bin/python -m analysis.look_write.extra_arms attn_s0 attn_s1 attn_s2 attn_s3 attn_s4
# 3. predictions P1-P4, verdict, results.json and tables.md (RESULTS.md = narrative + tables.md)
.venv/bin/python -m analysis.look_write.summarize
# exploratory (after the confirmatory part)
.venv/bin/python -m analysis.look_write.uov_tables mlp_s0 && .venv/bin/python -m analysis.look_write.fit_arms mlp_s0
.venv/bin/python -m analysis.look_write.uov_tables attn_s0 ... --band 8 63 --max-tokens 100000000
.venv/bin/python -m analysis.look_write.fit_arms attn_s0 ... --band 8 63
.venv/bin/python -m analysis.look_write.summarize
```

## Files

- `common.py`: constants (seeds, buckets, positions), weight access, exact forward pass with one head's pattern
  substituted, OV / TT / pure-pair score tables, final-LN scale.
- `selftest.py`: the checks of step 0 (writes `analysis/out/look_write/selftest.json`).
- `uov_tables.py`: the corpus pass (one weighted `np.bincount` per head, distance and chunk).
- `fit_arms.py`: base rule and arms (convex pattern cross-entropy, L-BFGS), output test, double-centred
  correlations.
- `extra_arms.py`: EXPLORATORY alternative ceiling forms (TT at d = 1..7; TT split by distance; TT including the
  self position, written but not yet run). Writes `<model>/arms_extra.json`, used by the sensitivity block.
- `summarize.py`: P1-P4 and the verdict exactly as pre-registered, plus exploratory sensitivity re-readings and
  paired SEs where per-window KLs exist.

Cache kept after the run: own-UOV tables for attn_s0..s4 (all heads, d = 1..7), d = 1 shuffled tables, N_1 recount.
The d >= 2 shuffled tables and mlp_s0's d >= 2 tables were deleted (rebuild with step 1).
