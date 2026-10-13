# ov02: what do model heads 0 and 2 write from earlier tokens? (attn_s0, open track, 2026-10-13)

## Assumptions

- Program = mechanism v1 (tag `mechanism-v1`), corpus tables from `analysis/mech_eval.stats()`.
- All fidelity numbers come from `analysis/vote_fit.py`. "Quick" means `--fit 16 --eval 32 --heads_windows 8`. "Full"
  means the defaults (fit 32 windows seed 2024, eval 64 windows seed 2000, head R^2 on 16 eval windows). The "program"
  R^2 baseline for heads 0+2 is 0.197 in quick mode and 0.214 in full mode. KLs are TRAIN windows, not harness splits.
- "OV map" = logits head h writes when it attends fully to token s: LN1(e_s + mean position) W_V^h W_O^h, final LN
  linearised at the typical scale 1.127, tied unembedding, rows centred. Insight only: no numbers from it enter any
  adoptable candidate.
- Weights were inspected for insight only. Every adoptable candidate uses closed-form TRAIN-corpus counts plus
  strengths fitted by vote_fit.

## 1. The gap splits into attention x votes, and votes are the larger part (insight)

Effect R^2 of heads 0+2 = pooled weighted R^2 of the model's earlier-token ablation effect, as in vote_fit.head_r2.
"Candidate alone" values come from quick runs. The candidates replace head0/head2 near+far.

| votes from earlier tokens \ attention | program rules | model's real attention |
|---|---|---|
| corpus skip-gram PMI (program as is) | 0.197 (whole program) | 0.212 (`ov02_oracle_pmi_real_attn`) |
| all corpus tables tried (skip, bigram, word-level, slot, copy) | 0.182 alone / 0.256 with program | 0.311 alone (`ov02_votes`, real, all) |
| model's exact OV map | 0.338 (`ov02_oracle_ov_prog_attn`) | **0.841** (`ov02_oracle_ov_real_attn`) |

- Correct attention with corpus votes barely helps: 0.197 -> 0.212.
- Exact votes with the program's attention reach only 0.338. Under the current attention rules, a perfect vote table
  could add about +0.14 R^2 at most.
- Only both together explain the heads (0.84). Even the best corpus vote tables under REAL attention stop at 0.31.
  So most of what heads 0 and 2 write is not a statistic of the token they read.
- KL side: exact OV under real attention improves the program only by 0.060 +- 0.004 (0.584 -> 0.524), with fitted
  strengths 0.47 / 0.46 instead of about 1. This is the ceiling for perfecting the earlier-token writes of heads 0
  and 2 inside the current program. The remaining 0.52 nats lie mostly elsewhere.
- The OV maps are high-rank. Truncated SVD (rows weighted by sqrt frequency, columns by sqrt unigram) under real
  attention gives effect R^2 of 0.124 at rank 2, 0.370 at rank 8 and 0.841 at rank 32. No handful of directions or
  features describes them.

## 2. What the OV maps look like (insight; s01, s02, s03)

Frequency-weighted mean row correlation of OV rows with corpus rows, over the 2000 v uniformly / weighted by
unigram probability. Rows are the tokens with count >= 20000.

| table | head 0 | head 2 |
|---|---|---|
| PMI_1 (bigram of the attended token itself) | **0.411** / 0.460 | **0.364** / 0.362 |
| PMI_2 | 0.369 / 0.282 | 0.286 / 0.277 |
| PMI_3 | 0.352 / 0.181 | 0.258 / 0.206 |
| RES_1 = PMI_2 - E_b PMI_1(b) (additive "beyond the current token") | 0.215 / 0.202 | 0.151 / 0.152 |
| CS1 conditional skip-gram (trigram counts, K = 30) | 0.298 / 0.257 | 0.323 / 0.327 |
| WPMI_1 / 2 / 3 (word-level skip-grams) | 0.371 / 0.289 / 0.265 | 0.338 / **0.372 / 0.373** |
| KPRED (same-slot affinity) | 0.125 / -0.125 | 0.163 / -0.024 |

- Cross-head OV row correlation: heads 0-2 0.154, 0-1 0.044, 2-3 -0.091.
- Global double-centred R^2 of OV on the corpus tables (rows weighted by frequency): head 0 0.131 with
  PMI_1+PMI_2+PMI_3+log P(v), and 0.156 with every table above. With p_uni weighting over v: 0.251 and 0.344. Head 2:
  0.085 -> 0.131 (uniform) and 0.124 -> 0.197 (p_uni). In the regression, COPY and KPRED come in negative (head 0
  copy -10.4, KPRED -3.4 to -4.6).
- Projecting the corpus rows onto the head's 32-dim output span raises the row correlation (head 0 PMI_1 0.41 ->
  0.56, PMI_2 0.37 -> 0.54; head 2 PMI_1 0.36 -> 0.52). Only 43-51% of a PMI row lies in that span. So part of the
  mismatch is that the head can only write 32 directions through the tied embeddings. Most of it is not.
- Examples (top / bottom of the OV map):
  - head 0, `' to'`: + `' do' ' play' ' wear' ' share'`, - `'ed' 'led' 'ing'`.
  - head 0, `' and'`: + `' went' ' played' ' were'`, - `' and' ' but' ' because'`.
  - head 0, `'.'`: + `'\n' <eos> ' One' ' But'`, - `'.' '?"' '?'`.

  Head 0 writes the attended token's own successors and votes against the token and words like it.
  - head 2, `' the'`: + `"'s" ' was' ' and' ' were' '.'` (what follows "the X").
  - head 2, `' a'`: + `' of' ' time' ' day' ' named'`.
  - head 2, `'.'`: + names (`' Ben' ' mom' ' Timmy' ' Lily'`).
  - head 2, `' Lily'`: - `' Lily' ' Anna' ' Timmy' ' Sara'`.

  Head 2 reads like a word-level skip-gram ("what comes after the next word") plus anti-repeat of names. For word
  onsets such as `' cr'`, head 2's map is the bigram (`'ack' 'own' 'unch'`), because head 2 attends to onsets mostly
  from the onset itself.

Interpretation: the OV map does not depend on distance. It is shared between the current-token part and the
earlier-token part. Head 0 spends about half its attention on the current token, where it does bigram work (together
with the direct path), so the map it writes from earlier tokens is mainly "what follows s". Distance-matched skip-grams
(the program's rule) are a cleaner corpus fact, but they are not what the head writes.

## 3. Why heads 0 and 2 are explained better together than apart (s05; exact head_r2 setting, 16 eval windows)

| quantity | value |
|---|---|
| R^2 head 0 / head 2 / both | 0.068 / 0.124 / 0.211 |
| corr(fitted part of head 0, fitted part of head 2) | **0.92** |
| corr(residual of head 0, residual of head 2) | **-0.26** |
| corr(head 0 effect, head 2 effect) | -0.15 |
| non-additivity: weighted SS of (y02 - y0 - y2) / SS(y02) | 0.014 |
| joint R^2 predicted from the single-head pieces | 0.213 (measured 0.211) |
| same, if the fitted parts were uncorrelated | 0.114 |
| same, if only the residuals were uncorrelated | 0.167 |

Head 0 fit: local0 0.21, local2 0.31, recall -0.29, phrase 0.17. Head 2 fit: local0 0.05, local2 0.79, recall -0.16,
phrase 0.15. Both: local0 0.28, local2 1.14, recall -0.47, phrase 0.33.

So "better together" comes from two effects:
1. The part the program explains is the same direction for both heads (0.92). It is a common skip-gram/phrase-like
   component, carried mainly by the LOCAL-2 direction. Summed, it grows in amplitude while the residuals grow only in
   variance: 0.114 -> 0.167.
2. The unexplained parts of the two heads partly cancel (correlation -0.26): 0.167 -> 0.211.

The two effects add almost exactly (non-additivity 1.4%). In logit space with the final LN linearised (s04, 16
windows), the earlier-token parts of the four heads are nearly uncorrelated: E0-E2 -0.09, E0-E3 -0.08, E2-E3 -0.07,
E0-E1 -0.09. var(E0+E2) / (var E0 + var E2) = 0.91. There is a little cancellation between heads, but no strong
cross-head cancellation. What the cancelling residual component is remains open.

## 4. Candidates (adoptable ones use program attention and corpus counts only)

| candidate | dirs | paired diff +- SE | R^2 heads 0+2 (program -> +cand; cand alone) | verdict |
|---|---|---|---|---|
| quick `ov02_votes` big (= bigram of earlier tokens) | 2 | -0.0041 +- 0.0004 | 0.197 -> 0.220; 0.048 | see full run |
| quick `ov02_votes` kpred,copy | 4 | -0.0034 +- 0.0009 | 0.197 -> 0.217; 0.029 | dead end alone |
| quick `ov02_votes` w2,w3 (word-level skip-grams) | 4 | -0.0008 +- 0.0004 | 0.197 -> 0.209; 0.124 | dead end |
| quick `ov02_votes` cs1 (conditional skip-gram, d = 1) | 2 | -0.0004 +- 0.0002 | 0.197 -> 0.205; 0.075 | dead end |
| quick `ov02_votes` big,kpred,copy,w2,w3 | 10 | -0.0094 +- 0.0012 | 0.197 -> 0.256; 0.182 | -> full runs |
| full `ov02_bigram_earlier` | 2 | -0.0038 +- 0.0004 | 0.214 -> 0.233; 0.046 | reject (below bar) |
| full `ov02_earlier_votes` | 10 | -0.0108 +- 0.0007 | 0.214 -> 0.255; 0.172 | passes the bar; prefer the compact form |
| full `ov02_earlier_votes4` | 4 | **-0.0102 +- 0.0006** | 0.214 -> 0.244; 0.119 | adopt (weak: +3 points of R^2) |

Fitted strengths:
- `ov02_earlier_votes4`: h0.bigram 0.61, h0.alike -4.21 (same-slot affinity, s excluded), h2.copy -5.80, h2.word3 0.035.
  h2.word3 is about zero; the lead can drop it and keep 3 strengths, but that 3-direction version was NOT run.
- `ov02_earlier_votes`: h0 bigram 0.55, slot -4.09, copy +3.99, word2 -0.05, word3 -0.33; h2 bigram 0.03, slot 0.15,
  copy -5.69, word2 0.08, word3 0.22.

In words:
- LOCAL-0 should also vote, from the tokens it reads, for their usual successors, and against words that fill the same
  slot as them.
- LOCAL-2 should vote against repeating the tokens it reads.

The gain (0.010 nats) is real and more than 15 SE. But it explains only 3 more points of the heads' effect. The
votes-under-current-attention ceiling is about +0.14.

Insight-only runs (never adoptable): `ov02_oracle_ov_prog_attn`, `ov02_oracle_ov_real_attn`,
`ov02_oracle_pmi_real_attn`, `ov02_oracle_lowrank` (ranks 2/8/32), and `ov02_votes` with OV02_ATTN=real:
- skip+big: 0.248 alone, KL -0.0033 +- 0.0016.
- all tables: 0.311 alone, KL -0.0096 +- 0.0017.
- skip+cs1: 0.229 alone, KL +0.0018 +- 0.0013.

## 5. Dead ends (with numbers)

- **Conditional skip-gram** (what the previous token adds beyond the current token's bigram, from trigram counts:
  sum_b P(b|s) [log P3(v|s,b) - log P1(v|b)]):
  - OV row correlation 0.30 / 0.32, below plain PMI_1.
  - Under program attention: -0.0004 +- 0.0002, R^2 +0.008.
  - Under real attention, alongside skip: no gain.
  - Smoothing K in {3, 30, 300} changes the row correlation by < 0.01.
- **Additive residual skip-gram** RES_d = PMI_{d+1} - E_{b ~ P_d(b|s)} PMI_1(b): OV row correlation 0.15-0.23, worse than
  raw PMI. (Not run through vote_fit.)
- **Word-level skip-grams** as extra votes for both heads: -0.0008 +- 0.0004. They match head 2's map better than PMI_2
  (0.37 vs 0.29) but add almost nothing to the output.
- **Bigram of earlier tokens alone**: -0.0038 +- 0.0004 (full), below the bar.
- **Anti-slot + copy alone**: -0.0034 +- 0.0009 (quick).
- **Fixing attention alone** (real attention, corpus PMI votes): R^2 0.212 vs 0.197. Better attention rules will not
  explain heads 0 and 2 unless the votes change too.
- **Low-dimensional description of the OV maps**: rank 8 explains 0.37 of the effect vs 0.84 at rank 32.

## 6. What this says about how the model works

1. Heads 0 and 2 each write a full 32-dimensional, token-specific table through the tied embeddings. They use the same
   table for the current token and for earlier tokens. The table is related to corpus statistics only loosely: row
   correlation about 0.4 with the attended token's bigram and about 0.3-0.37 with skip-grams. Even the best mix of
   corpus tables matches it at about 0.5.
2. Because the table is shared with the current-token part, head 0's writes from earlier tokens are mostly "the usual
   successors of that token" plus "not that token, nor a word like it". Head 2's writes lean towards word-level "what
   comes after the next word", plus anti-repeat. The program's distance-matched skip-gram is a cleaner statement of the
   corpus fact, but not what the heads compute.
3. Most of the heads' earlier-token effect needs the model's exact attention and the exact table together (0.84).
   Either one alone, with corpus stand-ins for the other, gives 0.21-0.34. The heads' contribution is a pair-specific
   (query x key) choice of which token to read, combined with a learned table. Simple attention rules times corpus
   statistics reproduce about a quarter of it.
4. Heads 0 and 2 share one explainable component (same direction, correlation 0.92) and have unexplained components
   that partly cancel each other (-0.26). That is why they are explained better together.
5. For fidelity, perfecting these writes inside the current program is worth at most about 0.06 nats.

## Files

- Scripts: `analysis/investigations/ov02/`:
  - `common.py`: OV maps and exact attention.
  - `s01_ov_vs_corpus.py`
  - `s02_ov_span.py`
  - `s03_word_tables.py`: word-level skip-gram and slot-kernel counts.
  - `s04_joint.py`: logit-space cross-head correlations.
  - `s05_joint_r2.py`: joint vs apart.
  - `summ.py`
- Candidates: `analysis/candidates/ov02_*.py`. Adoptable: `ov02_earlier_votes4.py` (recommended),
  `ov02_earlier_votes.py`, `ov02_bigram_earlier.py`. Exploration: `ov02_votes.py`. Insight only: `ov02_oracle_*.py`.
- Results: `analysis/out/candidates/ov02_*.json`. Tables and s01-s05 json: `analysis/out/candidates/ov02/` (244 MB).
- Resource note: the full vote_fit runs with 4 and 10 extra directions peaked at 3.7 GB and 4.6 GB RSS (`/usr/bin/time
  -l`), above the 3 GB guideline. The 2-direction run peaked at 2.7 GB. vote_fit's feature stack grows with the number
  of directions.
