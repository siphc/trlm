# Head 3 of attn_s0: what it does (investigation, prefix `head3`)

Assumptions: open track; all windows are TRAIN windows (`analysis.fitlib.train_windows`); exploration on seed 11
(as in the head-3 card), vote_fit on its default fit/eval seeds (2024 / 2000). "Effect" = model log-probs minus
log-probs with head 3's earlier-token part replaced by its mean (the harness 'earlier' ablation). R^2 = pooled
model-probability-weighted R^2 as in `analysis/vote_fit.head_r2`. Model-derived tables (U, model attention) are
used only as diagnostics, never in candidates. Nothing in program/, harness/, logs/ was touched.

Scripts (this directory): `h3lib.py` (exact numpy forward + head-3 earlier ablation + per-source logit votes),
`h3rule.py` (the card's attention rule, 27 scalars fitted to head 3's own pattern), `describe.py`, `diag.py`,
`diag2.py`..`diag5.py`, `peek.py`, `ov_svd.py`, `oracle_r2.py`, `tables.py`, `shared_r2.py`, `team.py`.
Outputs: `analysis/out/candidates/head3/` (JSON results; corpus tables `sub_left`, `sub_right`, `word_counts`,
`word_pmi`; diagnostic `U_model.npy`, `w_sd_model.npy`). Candidates: `analysis/candidates/head3_*.py`.

## 1. What head 3 does, seen in the data

Head 3's earlier-token effect is almost exactly linear in what it reads: the direct logit effect
`sum_{j<i} a_ij u_j / sdf_i - mean` (u_j = the logit vote of source j) leaves 3.5% of the effect's weighted
variance (24 windows). So "attention x per-source vote" is the right frame.

Where the effect sits (24 windows; share of the total shift KL(model || ablated) = 0.317 per position):

| query class | share | mean shift | self | d=1 | d=2-4 | d>=5 | main keys |
|---|---|---|---|---|---|---|---|
| LINKERS | 28% | 0.73 | .15 | .31 | .34 | .21 | the word(s) just before |
| SENTENCE_ENDS | 9% | 0.40 | .02 | .03 | .08 | .86 | SENTENCE_STARTS .24 |
| LINKING_VERBS | 6% | 0.52 | .19 | .31 | .38 | .13 | |
| word pieces (FRAGMENTS_*, SUFFIXES, LETTERS, ONSETS) | ~20% | 0.26-1.15 | .1-.55 | | | | the word's first token |

The largest single-position effects (describe.json) fall into three kinds:

1. **Word assembly.** A word piece looks at the first token of its word and the head votes for how that word
   ends: "frustr" reads ' fr' (0.93, d=2) -> "ated" +10.0 nats, "t" -9.9; "happin" reads ' happ' (0.77) -> "ess"
   +8.9; "excite" reads ' exc' -> "ment" +7.5; "excl" -> "a" +6.7; "sandw" reads ' sand' -> "ich".
2. **Parallel structure after linkers.** The linker looks at the word before it and the head votes for words of
   the same kind, and strongly against starting a new clause: "Mom and" -> ' Dad' +8.2, ' dad' +7.9 / ' he' -7.6,
   ' it' -6.8; "Lily and" -> ' Ben' +5.2, ' Tom' +4.7, ' Max' +4.5 / ' walked' -8.7, ' were' -8.2; "run, swing and"
   -> ' slide' +7.1, ' jump' +6.0, ' climb' +5.5 / ' she' -6.1, ' they' -6.1; "a big," -> ' scary' +6.9,
   ' bright' +5.6, ' long' +5.5; "a lot of" -> ' fun' +4.9; "From that" -> ' day'; "flew higher and" -> ' higher'.
3. **Linking verbs complete a frame**: "asked what was" -> ' happen' +7.0, ' wrong' +5.7; "His name was" -> names
   (' Jim' +5.8, ' John' +5.1) / ' a' -5.1, ' very' -4.8; "When it was" -> ' time'.

Nouns and sentence ends read the latest clause boundary; their votes are small and generic (see OV below).

## 2. Hypotheses and evidence

All per-source comparisons use pairs (query i, source j < i) with model attention a_ij >= 0.3 (1,717 pairs, 12
windows), comparing u_j with a corpus table f(s = t_j, v) over the vocabulary, weighted by the model's
next-token probabilities at i (the tokens that matter there) and by a_ij.

**H1. Votes are distance-matched skip-grams PMI_{d+1}(s, v)** (the card's rule, re-measured). Pooled R^2 of the
per-source votes: 0.102 (mean weighted correlation 0.29). By query group: LINKER 0.19, PIECE 0.11, VERB 0.07,
BOUNDARY 0.04, NOUN 0.03. Weak, as the card found. -> dead end (already known).

**H2. Votes are query-conditional: what s adds beyond the current token**, cond = log P(v | s at -d, q) -
log P1(v | q) from trigram (d=1) and 4-gram (d=2) counts. R^2 0.066 alone (correlation 0.297 vs 0.291 for PMI;
linkers 0.37 vs 0.37). Not better than PMI. -> dead end.

**H3. OV cannot see the distance, so the vote is a distance MIXTURE**, sum_d w_s(d) PMI_{d+1}(s, v) with w_s(d)
how often head 3 reads s at distance d (measured on 100 windows, diagnostic): log mixture 0.086, probability
mixture 0.113, earlier-only 0.104, vs 0.102 for the matched PMI. -> dead end.

**H4. Votes are about the next token's CLASS, not its identity.** Class-level skip-gram PMI (TARGET classes
0.034, ATTENTION classes 0.036), same-class indicators (0.000 / 0.001); combined with PMI 0.104. Only after
linkers does "same ATTENTION class as the word read" carry something (0.107 alone; with PMI 0.26). -> dead end
overall; a weak parallel-structure signal after linkers.

**H5. Parallel structure as substitution**: sub(s, v) = log sum_c P(c just before s) P1(v | c) - log P(v)
("v can stand where s stands", TRAIN bigram counts, closed form). Per-source R^2 alone 0.030, with PMI 0.134
(PMI alone 0.102); linkers 0.19 -> 0.25, noun queries 0.03 -> 0.11, verb queries 0.07 -> 0.13. Real but small.
Right-context substitution adds nothing (0.007). Within-word completion table (v 1-4 tokens later inside the same
word): word pieces 0.106 -> 0.121.

**H6. The bottleneck: votes or attention?** (oracle_r2.py, R^2 of head 3's effect, 12 windows)

| features | all | LINKER | PIECE | BOUNDARY | NOUN |
|---|---|---|---|---|---|
| exact direct logit effect (sanity) | 0.964 | 0.968 | 0.951 | 0.975 | 0.947 |
| exact attention x the model's own per-token vote table U (diagnostic) | 0.804 | 0.838 | 0.859 | 0.764 | 0.657 |
| card attention rule x U | 0.290 | 0.440 | 0.228 | 0.269 | 0.123 |
| exact attention x PMI_{d+1} | 0.076 | 0.156 | 0.081 | 0.041 | 0.006 |
| card attention rule x PMI_{d+1} | 0.040 | 0.119 | 0.026 | 0.039 | 0.000 |
| exact attention (d=1) x trigram conditional | 0.035 | 0.095 | 0.046 | 0.000 | 0.002 |
| program parts (LOCAL-0, LOCAL-2, RECALL, PHRASE) | 0.042 | 0.145 | 0.034 | 0.055 | 0.007 |

A token-level vote table (one row per source token, any position) would explain 80% with the exact pattern, so
the head is "attention x a per-token vote". Both factors are far from a readable rule: the card's attention rule
loses 0.80 -> 0.29 even with the right votes (word pieces lose most: 0.86 -> 0.23, because the rule penalises
word onsets as keys while the model reads them sharply), and corpus votes lose 0.80 -> 0.08 even with the right
attention. The vote table is the larger gap.

**H7. What the vote table is (insight only, ov_svd.py).** The weighted SVD of U (sources weighted by attention
received, outputs by unigram) has shares 0.27, 0.13, 0.10, 0.06, 0.05 for the first five components. Component 1:
sentence-initial subjects (' She', 'Once', ' He', end of story, ' said') vs '.', ' her', ' a', ' and', ','; output
'\n', ' her', ' She', ' He', ' the' vs '.', ',', ' and', 's', 'ing', 'ed'. Component 2: ' the', ',', ' a', ' was' ->
word pieces ('s', 'ing', 'y', 'ed') vs ' you', ' how', ' be' -> '.', ' and', ' the'. So the leading directions code
the kind of token that can come next (clause state: subject vs clause end vs word continuation), mixed with
specific associations (the card's ' run' -> ' jump', ' Lily' -> ' Ben'). No corpus table tried reaches more than
0.15 per-source R^2 (all features of diag2 together: 0.145).

**H8. Head 3 shares the local work with heads 0 and 2.** (shared_r2.py, team.py; 16 windows, eval seed.) Head 3's
effect is anti-correlated with heads 0+2's effect (weighted correlation -0.19 overall, -0.36 at noun queries,
-0.21 at word pieces, -0.17 after linkers); the three effects add almost exactly (non-additivity 1.8%). The
program's parts explain the SUM better than either part:

| heads ablated together | {3} | {0} | {2} | {0,2} | {0,3} | {2,3} | {0,2,3} | {0,1,2,3} |
|---|---|---|---|---|---|---|---|---|
| R^2 of program parts | 0.046 | 0.068 | 0.124 | 0.211 | 0.119 | 0.173 | 0.289 | 0.407 |

Explained weighted sum of squares: {3} 103, {0,2} 1,104, {0,2,3} 1,994 (of 2,244 / 5,232 / 6,898). Adding head 3
to heads 0+2 raises the explained mass by 890, eight times what head 3 alone gets credited with. Part of what
heads 0 and 2 write is cancelled by head 3, and what the three write together is closer to the program's
corpus-statistic votes (skip-grams, trigrams, 4-grams) than what any one head writes. After linkers head 3 is the
larger partner (sum of squares 594 vs 404 for heads 0+2).

## 3. Candidates (analysis/vote_fit.py, HEADS = [3])

Quick runs: --fit 16 --eval 32 --heads_windows 8; full runs: defaults (fit 32, eval 64, heads 16).
Paired difference = KL(candidate refit) - KL(base refit) (negative = better).

| candidate | rule | run | paired diff +- SE | head-3 R^2 program -> +candidate (alone) | verdict |
|---|---|---|---|---|---|
| head3_subst | card attention x sub(s, v), earlier tokens | quick | -0.00012 +- 0.00011 | 0.039 -> 0.045 (0.005) | reject |
| head3_subst_pmi | card attention x (sub, PMI_{d+1}) | quick | -0.00021 +- 0.00015 | 0.039 -> 0.047 (0.036) | reject |
| head3_linker | at LINKERS: previous word's PMI_2 and sub | quick | -0.00025 +- 0.00008 | 0.039 -> 0.054 (0.032) | reject |
| head3_linker | same | full | -0.00014 +- 0.00006 | 0.039 -> 0.059 (0.039) | reject (explains a little, no fidelity) |
| head3_onset | at word pieces: the word's first token's within-word completion PMI | quick | -0.0011 +- 0.0004 | 0.039 -> 0.040 (0.000) | reject |
| head3_onset | same | full | -0.00153 +- 0.00032 | 0.039 -> 0.039 (0.000) | reject (4.8 SE but far below 0.01; does not explain head 3) |

None reaches the bar (gain >= 0.01 for a new component). The linker candidate raises head 3's R^2 most
(0.039 -> 0.059 in the full run) without fidelity gain: PHRASE's trigram already gives the program "Mom and -> Dad" in KL terms.

## 4. Dead ends (with numbers above)

- Distance-matched skip-gram votes (re-measured): per-source R^2 0.10; with the card rule 0.04 of head 3's effect.
- Query-conditional trigram/4-gram votes (what the read word adds beyond the current token): 0.066 per source.
- Distance-mixed skip-grams (OV cannot see distance): 0.086-0.113.
- Next-token-class votes (TARGET / ATTENTION class skip-gram PMI, same-class indicators): 0.00-0.04.
- Substitution votes through head 3's attention rule (head3_subst): -0.0001 +- 0.0001 KL.
- Linker parallel structure (head3_linker) and word-onset completion (head3_onset): below the bar (table above).

## 5. What this says about how the model works

1. Head 3 reads one or two tokens with a query-dependent rule (word pieces read their word's first token,
   linkers and linking verbs read the word before them, nouns and sentence ends read the latest clause boundary)
   and writes a per-token vote that is a learned 32-dimensional code: partly "what kind of token comes next"
   (clause continues / clause ends / word continues), partly specific associations (substitutes such as
   Lily -> Ben, word endings such as ' fr' -> 'ated'). With exact attention, such a per-token table explains 80%
   of the head; the best corpus tables explain under 10%.
2. In the outputs it produces, head 3 is not a separate mechanism. It is the third member of the local-reading
   team with heads 0 and 2: its effect partly cancels theirs (correlation -0.19), and the program's local parts
   explain the three-head sum (29%) far better than head 3 alone (5%) or heads 0+2 (21%). The behaviour it adds
   (word completion, parallel structure, frame completion) is phrase knowledge that PHRASE's trigram and 4-gram
   counts already give the program, which is why no head-3 candidate gains fidelity.
3. Recommendation for the lead: in the consistency check and the README, compare LOCAL-0 + LOCAL-2 + PHRASE with
   the joint earlier-token ablation of heads {0, 2, 3} (R^2 0.29) rather than with heads 0 and 2 alone (0.21) and
   head 3 alone (0.05), and describe head 3 as "shares the local phrase work with heads 0 and 2, partly
   correcting them". A separate head-3 component built from corpus skip-gram-like tables is not supported.

## Surprises

- Head 3's effect is anti-correlated with heads 0+2's, and adding it raises the program's explained share of the
  local heads from 0.21 to 0.29.
- The card's attention rule loses most of head 3's effect even with the model's own votes (0.80 -> 0.29); word
  pieces read their word's first token, which the rule penalises.
- The conditional trigram (what the read word adds beyond the current token) matches the votes no better than
  plain PMI.
