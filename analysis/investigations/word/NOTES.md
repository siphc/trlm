# Investigation "word": word assembly in attn_s0 (open track)

## Assumptions

- Analysis windows are TRAIN windows (`analysis.fitlib.train_windows`, seed 2000 for evaluation unless noted, 2024 for
  fitting). No harness split was used. All KL numbers are KL(model || program), in nats per token.
- A token is a word piece if its text starts with a lowercase ASCII letter (`program.corpus.is_word_piece`). The
  word so far at position i is t_j0 .. t_i, where t_j0 is the latest non-piece token. L = i - j0 + 1.
  Position classes (`diag.classes`): **inside** (L >= 2: the current token is a piece of a word that started
  earlier), **onset** (L = 1 and cont > 0.5), **after multi-token word**, sentence end, other, position 0.
- Corpus tables written here (closed-form counts over the TRAIN stream, 35 s, peak 3.3 GB RSS), cached under
  `analysis/out/candidates/word/` (240 MB in total):
  - `wordstats.py` builds (last min(L, 4) tokens of the word so far, next token) counts, plus an "after a word" table.
  - `wordpairs.py` builds the V x V tables W2, WANY, WEND, WPIECE (docstring gives the counting rules).
- The analyses that use model weights (`ov_probe.py`, `ov_examples.py`, `decomp.py`, `votetable.py`,
  `lowrank.py`, `h1_*.py`) are for insight only. No number from them goes into a candidate, except attention-rule
  parameters fitted to head 1's own attention rows, which the rules allow.

## Where the word residual is (oracle_split.py, ledger_cls.py, diag.py; 64 windows)

| class | positions | KL | share of all KL | "word goes on?" (two-way) | which piece | which non-piece |
|---|---|---|---|---|---|---|
| inside (L >= 2) | 13.8% | 0.79 | 18.7% | 1.9% | 6.2% | 10.6% |
| onset | 6.7% | 0.72 | 8.3% | 0.3% | 7.5% | 0.5% |
| after multi-token word | 8.1% | 0.58 | 8.0% | 0.2% | 0.3% | 7.5% |

- Inside a word, most of the program's error is about **which word comes after the word ends** (10.6% of all KL).
  "Which piece comes next" is 6.2%, and "does the word go on" only 1.9%. The program gives continuing the word a
  little too little mass (P(next is a piece) at L=2: model 0.467, program 0.406).
- At word onsets almost all the error is **which first piece** (7.5% of all KL).
- At inside positions every mechanism helps. PHRASE helps most (removing it costs +0.34 there).

## Which heads read the earlier pieces of the word (attn_dump.py vs prog_attn.py; 32 windows)

Attention mass from a query inside a word with L = 2:

| head | self | word start | token before the word | program rule's mass on the word start |
|---|---|---|---|---|
| 0 | 0.36 | **0.32** | 0.10 | 0.18 |
| 1 | 0.02 | 0.04 | 0.03 | 0.05 |
| 2 | 0.20 | **0.24** | 0.19 | 0.20 |
| 3 | 0.28 | **0.23** | 0.09 | (not modelled) |

At L >= 3, heads 0, 2 and 3 also read the earlier pieces (0.10-0.23). At an onset, head 2 reads itself (0.81),
and head 0 reads itself (0.44) and the previous 1-3 tokens (0.36). So **heads 0, 2 and 3 all read the word start
from inside the word**, about equally. Head 2's rule says it "skips onsets", but it still reads the start of the
current word, and the fitted rule already gives that position about the right mass.

What each head writes when it reads a word start (ov_probe.py: median correlation, over 274 frequent onset tokens,
between the linearised vote and corpus tables, weighted by sqrt P(v)):

- head 0: like PMI_2 / W2 ("what sits two after the start when the word goes on", +0.37) and WEND (+0.30);
- head 2: like PMI_1 (the start's own first pieces, +0.38) and **against** WEND (-0.41), i.e. "the word goes on";
- head 3: like WEND ("what follows a finished word with this start", +0.28) and PMI_3 (+0.23).

In words: the heads split the work. Head 2 keeps the word open, head 0 predicts its next piece, and head 3
predicts the word after it. All of these correlations are modest, though.

## Hypotheses, evidence, verdicts

### H1. Heads vote for the corpus continuations of the whole word so far (candidate `word_prefix`). DEAD END

- vote = log Pw(v | word so far) - log P1(v | t_i), with Pw = (count + 30 P1) / (total + 30), and separate strengths
  for L = 2 and L >= 3.
- vote_fit (fit 32, eval 64): paired -0.00007 +- 0.00002. Fitted strengths 0.005 (L=2) and -0.001 (L>=3). Head R^2
  (heads 0, 2, 3) went from 0.2796 to 0.2805. Adding the vote by hand at inside positions makes things worse at any
  positive strength (KL there 0.858 at strength 0, 0.870 at 0.1, 0.99 at 0.25, 2.33 at 1.0; check_prefix.py).
- The corpus word-prefix distribution is **farther** from the model than the bigram (diag.py). At L=2,
  KL(model || Pw) is 2.28 / 1.94 / 1.59 / 1.26 for K = 0.3 / 3 / 30 / 300, against 1.83 for the bigram and 0.78 for
  the program. The model is less sure inside words than the corpus is (entropy 2.20 vs 1.91). Within a word the
  model does not know exact word identity. PHRASE (trigram and 4-gram, back-off 300) already holds as much word-prefix
  information as the model uses.
- Head-effect R^2 at inside positions for heads 0+2 (effects.py, 24 windows): program 0.187. The word-prefix
  direction alone explains 0.068-0.085, and program plus prefix gives 0.191-0.195.

### H2. Additive word-start and earlier-piece votes from corpus tables (effects.py, effects2.py). DEAD END

These directions are zero outside inside positions:

- onset_skip PMI_L(t_j0);
- word_skip, the sum of PMI over the earlier word tokens;
- ws_pmi1 PMI_1(t_j0);
- ws_wend WEND(t_j0);
- pc_pmi1 / pc_wpiece, read from the earlier pieces.

On its own each explains part of the inside effect of heads 0+2 (onset_skip 0.107, word_skip 0.095). On top of
the program, none adds more than 0.008 (h02 inside 0.187 -> 0.188-0.195; all five together 0.197; heads 0+2+3:
0.256 -> 0.267). The program's LOCAL readers already carry this naive-Bayes "word start votes skip-gram" signal.

- Candidate `word_end` (WEND(t_j0) at inside positions, 1 scalar; motivated by the head-3 and head-0 OV
  correlations): paired -0.0011 +- 0.0002, fitted strength -0.05, R^2 (heads 0, 3) 0.1054 -> 0.1055. Rejected.

### H3. The gap is in the vote tables, not the attention or the nonlinearity (decomp.py, votetable.py). INSIGHT (general, not only for words)

Target: one head's earlier-token effect (16 windows), measured as single-direction pooled weighted R^2.

| direction | head 0 all / inside | head 2 all / inside | head 3 all / inside |
|---|---|---|---|
| exact linearisation (real attention x real value vectors, real final-LN state) | 0.968 / 0.947 | 0.966 / 0.945 | 0.975 / 0.978 |
| same, constant LN scale (pure additive votes) | 0.885 / 0.886 | 0.920 / 0.923 | 0.929 / 0.917 |
| **program attention rule** x real value vectors | 0.247 / 0.404 | 0.444 / 0.406 | - |
| real attention x **corpus PMI_{d+1}** votes | 0.063 / 0.109 | 0.157 / 0.067 | 0.066 / 0.105 |
| program (rule x corpus votes) | 0.018 / 0.058 | 0.084 / 0.027 | - |

- Each head is a nearly linear, additive "read tokens, write a fixed vector per token" device: 89-93% of its effect
  with a constant LN scale. The final LayerNorm adds only about 5%.
- Replacing only the attention by the program's rule keeps 25-44% of the effect. Replacing only the votes by corpus
  skip-gram PMI keeps 6-16% (centred: 8-18%). **What a token writes when read is the main gap.**
- With real attention, the best corpus tables are: head 0 PMI_1+PMI_2+PMI_3+copy 0.100 (inside 0.198); head 2
  PMI_1..3 0.224; head 3 skip PMI_{d+1}+PMI_1+copy 0.091 (votetable.py, centred). The word tables (WPIECE, WEND)
  add ~0.005.
- Low-rank versions of the corpus tables (rank-8/32/128 SVD, plain or frequency-weighted; lowrank.py) are no
  better than the full tables, and mostly worse. The heads' votes are rank 32, but they are not the top singular
  directions of PMI. Dead end.

### H4. In-context word completion at onsets: head 1 recalls earlier pieces that can follow the current word start. MECHANISM CONFIRMED; candidate just under the KL bar

Evidence (onsets.py, 64 windows):

- 36% of onsets occurred earlier in the window. At those, the model puts **58%** of its mass on pieces that
  followed the same onset earlier, against the program's 29%. The model's top token is such a piece 77% of the
  time, the program's 53%.
- Examples: ' bu' -> 'l' (bulb) at 0.92 (program 0.19), ' tr' -> 'op' (trophy) 0.77, 'D' -> 'ave' 0.71,
  ' co' -> 'al' 0.72.
- The fingerprint is **successor affinity, not induction**. After 'trophies', 'S' -> 'op' gets 0.91 although no
  'S' was ever followed by 'op'. 'op' is simply an earlier piece that can follow 'S'.
- Head 1's attention at onset queries: 0.302 of its mass on earlier pieces (rule 0.154). Keys with
  P1(key | query) in 0.01-0.1 get 0.109 (rule 0.026). Keys that followed an earlier copy of the query get 0.069
  (rule 0.010). These overlap.

Attention term, fitted to head 1's OWN attention rows (h1_attn.py, h1_onset.py, h1_fit_word.py; 32 fit / 32 eval
windows). Pattern KL at onset queries / all queries:

| term | onset | all |
|---|---|---|
| none | 0.982 | 0.574 |
| induction only, B 0.278 | 0.966 | 0.574 |
| token successor max(0, PMI_1(t_i, t_j)), one global B 0.112 | 0.929 | 0.570 |
| successor gated by the onset indicator, B 0.377 | 0.854 | 0.566 |
| successor gated by cont(t_i), B 0.377 | 0.853 | 0.564 |
| **successor gated by start(t_i) = (1 - piece(t_i)) * cont(t_i), B 0.382** | **0.852** | **0.565** (inside unchanged 0.510) |

Adding induction to a successor term changes nothing (0.851).

vote_fit results (all of them replace RECALL's 15 vote directions by the same directions under the new attention):

| candidate | gate | paired diff +- SE | eval windows |
|---|---|---|---|
| `word_recall` | cont | -0.0058 +- 0.0016 | 64 |
| `word_onset_recall` | start | -0.0050 +- 0.0015 | 64, seed 2000 |
| `word_onset_recall` | start | -0.0040 +- 0.0008 | 96, independent seed 3001 |
| `word_onset_recall2` | start, plus a COPY_AT_START strength | -0.0050 +- 0.0015 | 64; extra strength fitted to -0.35, adds nothing; rejected |

Pooled over the two `word_onset_recall` evaluations the gain is about 0.0044, just under the 0.005 bar.
After the refit, RECALL's copy strength for WORD_ENDS_AND_LINKERS rises from 1.30 to 2.5-2.8. Word pieces are now
recalled from the right queries.

Head-1 R^2, like for like (r2_recall.py, 24 windows): one earlier-token RECALL direction, old attention vs new
attention, both with refitted strengths.

| candidate | onset | inside | other | all |
|---|---|---|---|---|
| `word_onset_recall` | **0.19 -> 0.33** | 0.28 -> 0.29 | 0.45 -> 0.45 | 0.426 -> 0.421 |
| `word_recall` (cont gate) | 0.19 -> 0.33 | **0.28 -> 0.25** | | 0.426 -> 0.416 |

`word_recall` is rejected because it hurts the inside class. The vote_fit tool's own head R^2 for these REPLACES
candidates (`word_recall`: 0.402 -> 0.373) cannot be compared like for like: it sets 15 directions that include
the self-attention part against one earlier-only direction.

Verdict: the mechanism is real and readable. Head 1 at a word start looks for earlier pieces that can follow it
and copies them. It is pattern-faithful (onset pattern KL -13%), and it raises head 1's explained share at onsets
from 19% to 33%. Its KL gain (0.004-0.005) is just under the adoption bar, and pooled head-1 R^2 falls by 0.005
because one coefficient now has to serve all position classes. It is a candidate like the end-of-story sink: adopt
it if the gate is relaxed for terms that fit the head's own pattern.

## What this says about how the model works

1. **There is no word assembly in the sense of looking up the whole word.** The model is less sure inside split
   words than the corpus. A corpus word-prefix vote is ignored by the fit and hurts if forced. Inside a word the
   model behaves like a product of a few one-token votes.
2. **The word start is read by three heads.** Heads 0, 2 and 3 each put about a quarter of their attention on the
   word start from inside the word. What they write from it differs: head 2 says "the word goes on" (the start's
   own first pieces, against word-final tokens), head 0 votes the next piece, and head 3 votes the word that
   follows. The program covers this jointly through FOLLOW + PHRASE + LOCAL readers. The remaining error is not in
   where the heads look but in what they write.
3. **What the local heads write is the main unexplained part, for every position class.** With the real attention,
   real value vectors and a constant LN scale, ~90% of each head's effect is reproduced. With real attention and
   corpus skip-gram votes, only 6-18%. The program's attention rules are a smaller loss (they keep 25-44%).
   Better corpus vote tables would pay off more than better attention rules. Low-rank PMI is not the answer.
4. **Head 1 completes words in context.** At a word start it looks for earlier word pieces that can follow the
   start (bigram affinity, not induction; a 1-layer model cannot do induction) and copies them. This is why
   'S' can be completed as 'op' after 'trophies'. Word onsets are the class where RECALL had the worst attention
   fit (pattern KL 0.98).

## Files

- Scripts: `analysis/investigations/word/*.py`. Each one's docstring says how to run it.
- Candidates:
  - `analysis/candidates/word_prefix.py`
  - `analysis/candidates/word_recall.py`
  - `analysis/candidates/word_onset_recall.py` (the near-miss)
  - `analysis/candidates/word_onset_recall2.py`
  - `analysis/candidates/word_end.py`
- Results:
  - `analysis/out/candidates/word_*.json` (vote_fit) and `word/vf_*.log`;
  - `analysis/out/candidates/word/{diag,effects,effects2,attn_dump,h1_attn,h1_fit_word,h1_fit_word_startcont}.json`.
