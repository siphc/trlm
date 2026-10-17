# Surprises

Record anything unexpected the moment it appears. Newest at the bottom.

## 2026-10-12 — Data format (phase 0/2)

- The parquet and `.txt` versions of the original TinyStories differ in whitespace. The `.txt`
  (stories separated by `<|endoftext|>`) collapses paragraph breaks `\n\n` to `\n` and carries a
  leading space/newline per story. The parquet keeps `\n\n`. We use the parquet as the source.
- The corpus contains cp1252 mojibake of UTF-8 punctuation: in the validation split, `â€` occurs
  7,144 times (`â€œ`, `â€™`, ...), against 108 correct `“`. AGENTS.md says keep raw casing and
  punctuation, so we do not repair it. The byte-level BPE will spend some vocab on these bytes.
- The first validation story starts mid-sentence ("Spot. Spot saw the shiny car..."). Some stories
  are truncated. Not repaired.
- `huggingface_hub.snapshot_download` stalled on the train parquet files; plain `curl` on the
  `resolve/main` URLs worked.

## 2026-10-12 — Duplicate stories (phase 2)

- 6,213 rows of train-minus-dev are byte-identical to some dev story, and 6,553 are identical to some
  test (official validation) story (`logs/duplicates.json`). Dev rows are excluded by index, so the
  tokenizer and model never see the dev *rows*, but they do see identical text from other rows.
  Fidelity is measured against the model, so this mostly affects data-loss numbers (model and
  n-gram cross-entropy on dev/test are slightly optimistic), not KL.
- `\n\n` paragraph breaks tokenize as two separate `\n` tokens (the GPT-2 byte-level regex splits
  a whitespace run before a letter), so the BPE never merges them.

## 2026-10-12 — KL direction vs cross-entropy (phase 1 self-test)

- On a 10M-token sanity model, the bigram baseline is only 0.09 nats worse than the model in data
  cross-entropy, yet KL(model || bigram) = 1.19 nats. The bigram's Dirichlet smoothing gives
  unseen pairs ~1e-10, and the (undertrained, diffuse) model puts mass there. KL(model || program)
  punishes any token the program rules out that the model does not. Programs need a floor.

## 2026-10-12 — Throughput

- On the M4, `torch.set_num_threads(10)` cuts CPU training throughput from 137k to 37k tokens/s
  (efficiency cores). The torch default (4 threads) is best. MPS gives ~250k tokens/s, which makes the
  whole pilot plus 10 final runs feasible on the laptop in a few hours.

## 2026-10-12 — Pilot LR check

- Dev loss falls steeply with LR at 12.5M tokens: 3.670 (5e-4), 3.233 (1e-3), 2.917 (2e-3). The best LR is
  at the top edge of the AGENTS.md grid, so the true optimum is probably higher. Protocol followed
  (2e-3 chosen); one extra explore point at 4e-3 is run after the ladder for information only.
- The first two LR runs took 352 s and 702 s, the third 74 s (same config, same device). The slow ones
  ran while the machine was otherwise idle waiting on a human answer; likely macOS throttling/App Nap
  of a background process. Wall time is not a result here, but throughput numbers can be unreliable.

## 2026-10-12 — Vocabulary structure (pre-phase 5, data only)

- 106 of the 2000 tokens never occur in the train stream (byte tokens fully absorbed by BPE merges).
  The model can only learn to push their logits down; the program can treat them as one category.
- Ward clustering of train-bigram contexts gives clean word classes (prepositions, modals, past-tense
  verbs, names, determiners, emotions, colours, word-onset fragments, byte fragments).
- Class structure alone explains under half of the bigram: KL(data bigram || class->class) = 1.38 nats
  vs 2.47 for unigram; adding the top-32 exact continuations per token brings it to 0.22-0.26.

## 2026-10-12 — attn_s0 first look (phase 5)

- Zero-ablating one head of attn_s0 costs up to 3.9 nats (head 0) and all four heads 35.6 nats of KL.
  Head outputs have norms 3.5-7.6 against 2.2 for the token embedding, so the final LayerNorm sees a
  completely different vector. Zero ablation is meaningless here; the consistency check needs mean
  ablation. The harness (built in phase 1) only does zero ablation. Raised with the human.
- Heads 0 and 2 put ~50% / ~34% of attention on the current token itself: most of the model's "bigram"
  goes through these heads, not through the direct embedding path.
- Final LayerNorm scale varies 0.57-2.13 across positions (mean 1.13, sd 0.24): a ~4x range in effective
  temperature that an additive program does not have.
- Harness self-test with the model wrapper failed the causality check on attn_s0: float32 log-probs for a
  5-token prefix differ from the full-window run by 2.9e-5 (> 1e-5 tolerance) because of different kernel
  shapes. In float64 the difference is 2.5e-14. Fixed the self-test fixture to run in float64; scorer
  unchanged.

## 2026-10-12 — Resume bug on MPS (phase 4)

- Resuming attn_s1 on MPS crashed: `torch.load(..., map_location="mps")` moves the saved CPU RNG state to
  the GPU and `torch.set_rng_state` rejects it. The resume sanity test only covered CPU. Fixed (`.cpu()`),
  and the test now runs on MPS too (weights agree within 1e-4).

## 2026-10-12 — Compression (phase 5)

- The pruning *ranking rule* was the dominant problem, not the schedule: first-order "data importance" (linear in
  attention mass) overvalued the diffuse head 1 by ~1/a; the second-order Fisher cost 0.5 w^2 sum a^2 q(1-q) took
  60k exceptions from 0.844 to 0.514 (v6 -> v7).
- The tables have exact gauge freedoms (row constants, column/class/diagonal effects). Re-centring them losslessly
  before pruning is worth 0.58 nats at a no-refit cut, 0.12 after a short refit, and 0.019 after the full gradual
  schedule at 60k — but 0.045-0.049 at 256k-1.1M, where 1.78M entries match the dense student.
- The same add-back swap protocol flips sign between v8 (-0.0068) and v9 (+0.0017).
- dev_fast tracks held-out TRAIN KL within +0.002 to +0.003 for v7-v10 (v4/v5: +0.01).
- Machine pressure: the disk filled (7.2 GB of per-round 160 MB checkpoints, plus agents' scratch); now saved
  sparsely (~1 MB per pruned student).

## 2026-10-12 — Role-split classes (v11)

- Splitting the one taxonomy into separate CONTEXT (rows) and TARGET (columns) classes was worth 0.013 nats at 60k
  exceptions, but only 0.002-0.004 at 0.5-10M exceptions: the gain over v10 first shrinks and then grows as the
  budget falls, so better class defaults matter most in exactly the regime the main line operates in.
- Refitting with new classes moved the heads' measured roles in magnitude (head 2's removal cost 0.27 -> 0.19, head 3's
  0.16 -> 0.21, FOLLOW exceptions 1.43 -> 0.89): 10 of 32 claims needed new numbers. Claims are tied to one fit.

## 2026-10-13 mechanism v1 consistency check (attn_s0)

- The program's head parts move the output 5-10x less than the model's heads (earlier-token removal: program 0.04-0.08
  nats, model 0.41-0.49), although the program as a whole is 0.58 nats from the model and its parts point the right
  way (agreement 0.27-0.64). Most of what each head does from context is not in the program; output KL alone hid this.
- Heads 0 and 2 are explained better together (R^2 0.21) than apart (0.07 and 0.12): the program's split of local
  reading between LOCAL-0 and LOCAL-2 is not the model's split.
- Not cancellation: the heads do not carry large opposing copy votes that net out (analysis/cancellation.py).

## 2026-10-13 discovery round 2 (attn_s0; logs/mech_v2_investigations.md)

- Heads 0, 2 and 3 are each about 90% additive "read a token, write a fixed vector" devices, yet corpus skip-grams with
  the real attention explain only 6-16% of what each writes; all corpus tables together 31% for heads 0+2.
- Head 3's effect is anti-correlated with heads 0+2's (-0.19; -0.36 at noun queries); the program explains the three
  heads jointly (R^2 0.289) far better than head 3 alone (0.046). After linkers head 3's effect exceeds heads 0+2's.
- A 2-scalar "own-vote usefulness" attention term beats the exact token-token query-key term in exact-model output KL
  for heads 0 and 2; another head's vote predicts nothing. Where a local head looks follows from what it writes.
- Fitting head 1's attention pattern better (key-salience flags) made the program's output worse: the pattern weights
  every key alike, while the output depends only on keys whose votes matter.
- RECALL's "1.6x too weak" is attention diffuseness: with head 1's real attention its votes fit at coefficient 0.92.
- Inside split words the model is less certain than the corpus given the whole word-so-far.
- vote_fit's head-R^2 check understates candidates that REPLACE a part, by about 0.07, because the 'program+candidate'
  set then includes current-token votes, which the earlier-token ablation leaves untouched. Fix before relying on it
  for replacement candidates.

## 2026-10-14 look-follows-write replication (logs/look_follows_write_results.md)

- In one head (attn_s1/h3) the exact token-token query-key term makes the output worse when added to a generic
  attention rule (-0.052 +- 0.011), while own-vote usefulness helps (+0.092). Same for near-threshold head s2/h0 (-0.113).
  The "ceiling" is not a ceiling there; likely because the term is not applied at the self position (65% of that
  head's attention), untested.
- In the program, head 2's key-informativeness weight halved once own-vote usefulness was added: the law absorbed an
  older hand-made attention term.

## 2026-10-14 deep study, development seed

- The 2-block attention-only TinyStories model (attn2L_s0, dev loss 2.275) has no induction head by the standard test
  (attention from the second copy of a repeated random sequence to the token after the earlier occurrence: at most 0.03).
  Its far-looking layer-1 head (mean distance 44 tokens) does not copy that way.

## 2026-10-17 deep study results

- Last-layer heads of 2-block models follow "look where your write is useful" more strongly than the one-layer heads
  (coupling coefficients 0.6-0.8 vs 0.25), while layer-0 heads of the same models, at the same depth as the one-layer
  heads, barely follow it. Writing to the output, not depth, is what matters.
- The exact gradient through the final LayerNorm (TOTAL) predicts attention only half as well as its typical-scale
  linearisation (LTOTAL): the per-position LayerNorm Jacobian hurts the prediction.
- Head pairs cancel each other's effects in one-layer models (all 5 seeds), never in 2- and 4-block models, even within a
  layer of the same width.

## 2026-10-17 goal 2 (readable copy), autonomous session (logs/goal2_plan.md)

- **Seeds of the same network are far apart.** On 64 TRAIN windows, attn_s1..s4 are 0.38-0.42 nats from attn_s0
  (KL(attn_s0 || seed k)); the average of seeds 1-4 is 0.25 away; mlp_s0 0.58. The program (v1.2) is 0.56 away, so a
  re-trained copy of the very same network is only about 30% closer to attn_s0 than the program is.
- **The program is equally far from every seed** (0.557-0.574), although it was fitted to attn_s0 only, and its data
  cross-entropy (2.501) equals attn_s0's (2.506). It is a seed-agnostic description: refitting all its scalars to
  attn_s1 made its held-out KL to attn_s1 slightly worse (0.561 -> 0.569).
- **The model's current-token table is mostly per-token, and partly seed-specific.** With every head's earlier-token
  part exact, a context-free per-token table reaches KL 0.07 where the program's corpus directions reach 0.33. Of that
  table's variance (model-mass weighted), the corpus bigram explains 0.60-0.72, the mean of the other four seeds 0.83-0.87:
  about 15% is seed-specific, about 20% shared by all seeds but not a corpus statistic tried.
- **RECALL needs a saturation, the other mechanisms do not.** Dividing RECALL's summed vote by sqrt(1 + c spread^2)
  (the final LayerNorm's limit on one large write) gains 0.016 nats; the same for LOCAL-0/LOCAL-2 0.002, PHRASE 0. With
  it, RECALL's fitted strengths grow about fivefold: its vote acts almost like a fixed-size vector whose direction the
  attention chooses.
