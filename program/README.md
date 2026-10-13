# Mechanism program for `attn_s0`

A NumPy program that predicts the next-token distribution of `attn_s0`, a one-block, attention-only TinyStories
language model (4 heads, vocabulary of 2000 tokens), without using the model's weights. Entry point:
`predict(tokens)` in `__init__.py`; row `i` of its output is the distribution after `tokens[: i + 1]`.

**How it explains the model.** Every number that is specific to a token or a pair of tokens is a statistic of
the training stories, counted when the program starts (`setup(corpus)`, about two minutes; the harness hands
the program the training token stream, a one-time exception approved in `logs/decisions.md` item 7). The model
is used only to fit a few dozen named scalars: how strongly each mechanism votes, and the parameters of each
head's attention rule, which were fitted to *that head's own attention pattern*, not to the output. So the
program says *what* each head reads and *what* it does with it in terms of the language itself; it contains no
table learned from the model.

Fidelity (harness, KL(model ‖ program) in nats per token): **0.589** on dev_fast (200 windows); the program's
top token is the model's top token 62% of the time; it recovers 98.7% of the cross-entropy gap between a corpus
bigram and the model. For scale: the corpus bigram alone is 1.20 nats from the model, and the earlier
distillation program (tables fitted to the model, git tag `distillation-v11`) reached 0.452.

## The algorithm, top-down

At every position `i` each of the 2000 possible next tokens `v` gets a score: the sum of the votes of five
mechanisms. A softmax turns the scores into probabilities.

```
score_i[v] = FOLLOW(t_i, v)                          follow.py   the current token's usual successors
           + LOCAL-0  reads the last few tokens        head0.py    [model head 0]
           + LOCAL-2  reads informative recent words   head2.py    [model head 2]
           + RECALL   recalls earlier story words      head1.py    [model head 1]
           + PHRASE   continues known phrases          phrase.py   [behavioural; parts of heads 0 and 2]
```

1. **FOLLOW**: `0.50 · log P1(v | t_i) + 0.37 · log P(v)`, the corpus bigram. In the model this is the sum of
   the direct path and of heads 0 and 2 looking at the current token.
2. **LOCAL-0 and LOCAL-2** (the local readers). Each spreads attention `a(i, j)` over the current and earlier
   positions by a short rule, and each token it reads votes for *what the stories put at the next position
   given that token and how far back it is* (corpus skip-gram PMI; `readers.py`). LOCAL-0's attention falls
   steeply with distance (odds about ×0.75 per token back), depending only on the spelling of the current
   token. LOCAL-2 prefers sentence ends and words that predict a lot, and skips word pieces and word onsets.
3. **RECALL**. Attention decays slowly (half-life about 23 tokens), favours names and content words, and
   favours earlier words of the kind that usually follows the current word (after `' the'`, nouns). Each
   recalled word votes to be repeated, for its topic (words that tend to appear 9 to 24 tokens after it in the
   stories), and slightly against its own usual successor.
4. **PHRASE**. Corrects the bigram with trigram and 4-gram counts of the stories. It stands in for the part of
   heads 0 and 2 that their simple attention rules miss: token-pair-specific choices of where to look, mostly at
   the previous one or two tokens.
5. Add the votes and normalise (`log_softmax`). Nothing else happens.

Model head 3 is not modelled (see "What is not explained").

## What each part contributes

Measured with `analysis/mech_eval.py` on 64 TRAIN windows (not the harness splits): KL(model ‖ program) with
everything on is 0.582.

| part | one-sentence role | KL rise when removed |
|---|---|---|
| FOLLOW | the current token's usual successors (corpus bigram) | +1.29 |
| PHRASE | trigram and 4-gram corrections to the bigram | +0.17 |
| LOCAL-0 (head 0) | the last few tokens vote for what follows them at that distance | +0.094 |
| RECALL (head 1) | earlier names and content words vote for themselves and their topic | +0.086 |
| LOCAL-2 (head 2) | informative recent words and sentence ends vote the same way as LOCAL-0 | +0.076 |

Standard errors of the rises are 0.002 to 0.008. The context mechanisms overlap: FOLLOW on its own is 1.70 nats
from the model with its strengths as fitted here (1.20 if refitted alone), so the four context mechanisms
together remove 1.12 nats, while their single-removal rises sum to only 0.43.

## Where the numbers come from

- `corpus.py` counts everything token-specific from the training stream: unigram and bigram, skip-gram pairs
  at distances 2 to 8 within a story, a topic table, a class successor table, trigram and 4-gram counts, and
  token features (how much a token tells about the token `d` places later, how often a word piece follows
  it, spelling flags). Its docstring lists the exact counting rules.
- `categories.py`: three systems of hand-named word classes, found from the training text alone (no model).
- `vocab.py`: the 2000 token strings of the tokenizer.
- Named scalars, each with a comment saying how it was found:
  - 28 attention-rule parameters (head 0: 8, head 2: 12, head 1: 8), fitted to the matching model head's
    attention rows (cross-entropy between the rule's and the head's attention, 32 TRAIN windows);
  - 25 vote strengths (FOLLOW 2, head 0: 3, head 2: 3, head 1: 15, PHRASE 2) and the 3 smoothing numbers of the
    bigram rule, fitted to the model's log-probs with the attention rules frozen;
  - a few counting choices made by hand (skip-gram distances up to 8, topic window 9 to 24 tokens, the
    phrase back-off weight 300 chosen from {3, 30, 300}).

## What is not explained

- **Head 3** (clause context: it carries a phrase across linking words such as `' and'`, `' of'`, `' to'`).
  Readers built from corpus statistics gained nothing (`logs/dead_ends.md`).
- **Inside split words**: completing a word whose first pieces have been seen is the largest single share of
  the remaining KL (about 27%).
- **Pair-specific attention** of heads 0 and 2 (which earlier token to read given the current token) is
  covered only behaviourally, by PHRASE.
- The remaining KL by part and position class is in `logs/mechanism_cards.md` and `logs/mech_v1_report.md`.
