# Status: phases 3–5 (stopped on battery power)

Date: 2026-10-12, 15:45. Written when the laptop was running on battery (19%, discharging).

## Assumptions (read first)

1. Training hardware is the laptop's Apple M4 GPU (MPS, fp32), approved by the human; `configs/frozen.yaml` v2.
2. Pilot followed AGENTS.md section 5 literally (LR picked at 12.5M, grid 5e-4 / 1e-3 / 2e-3).
3. Discovery fits use TRAIN windows only (seeded draw of 6,000 windows); dev_fast is used only through the
   harness. Probe samples for attn_s0 were being generated when work stopped.
4. Program tables are fitted by regression to the model's log-probs and indexed by token text and by named
   word classes (each literal ≤ 64 numbers). Word classes come from train-data statistics, not from a model.

## Phase 3 — pilot (done, `logs/pilot.md`)

Budget: **one epoch, 527,482,880 tokens, peak LR 2e-3** (largest rung wins; 200M is 2.5% worse). The LR grid's
top edge wins; an extra 4e-3 point is only 0.48% better at 200M, so it was not adopted.

## Phase 4 — final models (1 of 10 done, paused)

| model | dev_full loss | notes |
|---|---|---|
| attn_s0 | 2.5095 | 3,174 s; the last ~1,200 steps were very slow (see below) |
| attn_s1 | — | paused at step ~650 (resumes from `latest.pt`) |
| others | — | not started |

Training slowed ~40× around 14:30 when the laptop went to battery and a CPU-heavy probe-sampling job started
next to it. A 15 s throughput test on battery still ran at full speed (307k tokens/s), so the cause is likely
background-process throttling plus CPU contention, not raw GPU speed. Resume with:

    nohup caffeinate -dimsu .venv/bin/python -m src.train.runs final --device mps \
        --budget_tokens 527482880 --peak_lr 0.002 > logs/runs/final_runs.out 2>&1 &

(~30 min per attn model, ~35 min per mlp model on mains power; completed runs are skipped.)

## Phase 5 — discovery on attn_s0 (started)

### What the model does (open-track analysis)

- Heads 0 and 2 are local: about 49% / 34% of their attention is on the current token itself, most of the rest
  on the previous 1–3 tokens. Their output norms (7.6, 6.7) dwarf the token embedding (2.2), so most of the
  "bigram" is computed by these heads attending to self, not by the direct path.
- Head 1 is long-range and diffuse (mean distance 26, entropy 3.4 nats): story-level context.
- Head 3 is mid-range (mean distance 7.8).
- The final LayerNorm scale varies 0.57–2.13 across positions (a ~4× effective-temperature range).
- The best possible context-free predictor (the model's mean prediction given the current token) has
  KL 1.09 on train windows. Everything below that must come from attention.

### Fidelity–complexity curve so far (dev_fast, 200 windows; all rows in `logs/iterations.csv`)

| version | program | KL(model ‖ program) | top-1 | numbers in tables (approx.) |
|---|---|---|---|---|
| v0 | uniform | 5.066 | 0.005 | 0 |
| v1 | WORD (model's mean prediction) | 3.471 | 0.107 | 2k |
| v2 | + 64×64 class bigram | 2.424 | 0.196 | 6k |
| v3 | + 20k token-pair FOLLOW exceptions | 1.348 | 0.366 | 26k |
| **v4** | **+ 4 heads, class-level attention / recall / copy** | **1.170** | **0.480** | **~60k** |
| v5 | dense exceptions, global 40k budget (dead end) | 1.190 | 0.536 | ~70k |

Reference points on the same windows: unigram 3.472, bigram baseline (4M-entry table) 1.339, trigram 1.219,
model wrapper 0.000.

### The central finding

A dense student with exactly the program's structure (FOLLOW table + four heads with salience / class
preference / recency attention and per-token RECALL votes) reaches **KL 0.247** on held-out train windows.
The structure is expressive enough. The loss comes almost entirely from compressing millions of small
token-pair entries into a few tens of thousands of readable ones (0.25 → 1.17). Two pruning schemes are in
`logs/dead_ends.md`.

### Next steps (in order)

1. **Gradual pruning during the fit** (shrink the exception budget over many refit rounds instead of one cut).
2. **A confidence term for the final LayerNorm**: inverse temperature from how concentrated each head's
   attention is (sum over distinct tokens of squared attention mass). Few constants, readable, and it should
   absorb many exceptions.
3. **Group-structured exceptions**: keep whole (token, class) blocks or "concepts" (trigger list → vote list),
   which read as rules ("after seeing ' park', head 1 votes for ' slide', ' swing', ...").
4. Finer, still-named word classes (two-level taxonomy) to strengthen the class-level default.
5. Rename heads by role once the claims are measured; write CLAIM docstrings and claim tests.

## Questions for the human

1. **Power**: please plug the laptop in so the remaining 9 training runs can finish (~5 h).
2. **Ablation in the harness**: zero ablation of one head costs up to 3.9 nats of KL and all heads 35.6, so the
   section 9 consistency check is meaningless with zero ablation. May I switch the harness's ablation (used only
   by claim tests) to **mean ablation** (head output replaced by its mean over dev positions)? Editing
   `harness/` needs your approval.
3. **Readability direction**: the most faithful programs will carry tens of thousands of token-pair entries in
   tables (each literal ≤ 64 numbers, keyed by token text). Is that acceptable, or should I favour fewer,
   rule-like "concept" groups at some cost in KL?

## How to resume (after charging)

State at stop: all processes stopped; repo committed; `program/` holds v4 (dev_fast KL 1.1696).
attn_s0 is final; attn_s1 has a resume checkpoint at step ~650; probe samples exist for attn_s0 only.

1. Training (skips finished runs, resumes attn_s1):

       nohup caffeinate -dimsu .venv/bin/python -m src.train.runs final --device mps \
           --budget_tokens 527482880 --peak_lr 0.002 > logs/runs/final_runs.out 2>&1 &

2. Probe samples as models finish (keep it off the CPU while training: it uses 8 threads):

       nohup scripts/training/probes_as_models_finish.sh > logs/runs/probes_build.out 2>&1 &

3. Discovery on attn_s0 continues with gradual pruning (next step 1 above); fits run on MPS and share the GPU
   with training, so expect both to slow down.

Also tried before stopping (negative): predicting the final-LayerNorm scale from attention concentration,
R^2 = 0.18 (`logs/dead_ends.md`).
