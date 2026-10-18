# Logs index

Reports in reading order within each group. Raw run outputs (`.out`, `.log`, run JSON) are in `logs/runs/`.

**Running records**
- `decisions.md`: every human decision, numbered; read first.
- `iterations.csv`: one row per harness run (written by the harness). `dead_ends.md`: every failed hypothesis with its
  evidence. `surprises.md`: unexpected findings as they appeared. `human_judging.md`: the judging sheet.

**Phases 0-4: setup, training, pilot**
- `phase0-2_report.md`, `pilot.md`, `phase3-5_status.md`, `phase4_report.md`, `throughput.md`; `param_counts.json`,
  `param_counts_deep.json`.

**Replacement programs for attn_s0**
- Distillation era (tables fitted to the model; ended by decision 5, git tag `distillation-v11`): `consistency.md`,
  `compression_experiments.md`.
- Mechanism program ("explain, don't fit"): `mechanism_cards.md` (one card per head), `mech_v1_report.md` (v1),
  `mech_v2_investigations.md` (round 2), `mech_v3_report.md` (round 3: v1.2 -> v1.3, how far apart seeds are).
- `token_geometry_results.md`: named properties versus the models' shared token geometry; tables versus attention.

**Study: "where a head looks follows from what it writes"**
- One layer: `prereg_look_follows_write.md` -> `look_follows_write_results.md`; addenda
  `prereg_look_follows_write_addendum.md`, `prereg_look_follows_write_addendum2.md` ->
  `look_follows_write_addendum_results.md`.
- Deeper models: `deep_study.md` (running log and decision rules), `prereg_deep_look_write.md` ->
  `deep_look_write_results.md` (with the independent verification).
- Pre-draft for outside readers: `../writeups/look_follows_write_predraft.md`.

**Next studies (decision 10)**
- `plan_attention_law_and_readability.md`: fixed-vote models, training against the look-follows-write law, readability
  and the arbitrariness between training runs; the readable-training side study's running log is
  `readable_training.md`.

**Black-box track**
- `blackbox_track.md` (started, not continued); `blackbox_queries.jsonl` (query log written by the server).
