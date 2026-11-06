# Black-box track (AGENTS.md section 10)

Started 2026-10-12 ~23:40 for attn_s0.

- Repository: /Users/svi/Desktop/trlm_blackbox, created by `scripts/setup_blackbox.py`: AGENTS.md, configs/, data
  (symlinks to this repo's token streams, evaluation sets, tokenizer, train-split n-gram tables), the harness
  (patched so the model is reached only through the logits server; no ablation), src/train/data.py, the n-gram
  baselines, scripts/iterate.py, program/ with only vocab.py, BLACKBOX.md. No models/, no analysis tools, no
  open-track program, logs or notes. Fresh git history.
- Model access: `python -m blackbox.server --model attn_s0` runs from this repo (key in logs/.blackbox_key,
  gitignored); every query is counted in logs/blackbox_queries.jsonl.
- Session: a separate agent with no shared context; its prompt contained AGENTS.md's pointers, its repo layout and the
  query key only — nothing about the open track's structure or results.
- Isolation limits (honest statement): programs scored by the black-box harness run under sandbox-exec, which also
  blocks this repo's models/, analysis/, program/, logs/ and data/. The black-box agent's own tools are not
  sandboxed; it is instructed not to read outside its repository. The server never sends weights.
