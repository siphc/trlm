# Human judging sheet (AGENTS.md section 12)

For the mechanism program (program/, v1.3 at the time of writing). The earlier sheet was for the distillation program
(git tag `distillation-v11`), whose files (`attention.py`, `head3.py`, `tables/`) no longer exist.

Read `program/` top-down, blind to the metrics if possible (program/README.md quotes fidelity numbers in its second
paragraph and in "What each part contributes"; skip those). Suggested order: `README.md` (algorithm section), then
`__init__.py`, `follow.py`, `head0.py`, `head2.py`, `readers.py`, `head1.py`, `phrase.py`, then skim `corpus.py` (how the
statistics are counted) and `categories.py` (hand-named word classes). For each component answer the four questions
with 0 (no), 1 (partly) or 2 (yes), and write concrete rewrite requests for anything scored below 2. The agent turns
each request into a change.

Questions:
1. Can I state in one sentence what it does?
2. Can I predict what happens to the output if it is removed?
3. Do names and comments match the behaviour?
4. Is any table or constant opaque to me? (2 = nothing opaque)

| component | Q1 one sentence | Q2 predict removal | Q3 names match | Q4 nothing opaque | rewrite requests |
|---|---|---|---|---|---|
| `__init__.py` (the algorithm, top-down) | | | | | |
| `follow.py` FOLLOW (current token's usual successors) | | | | | |
| `head0.py` LOCAL-0 (last few tokens, steep recency) | | | | | |
| `head2.py` LOCAL-2 (informative recent words) | | | | | |
| `readers.py` (shared vote rules and own-vote usefulness) | | | | | |
| `head1.py` RECALL (earlier story words, fixed-size vote) | | | | | |
| `phrase.py` PHRASE (trigram and 4-gram corrections) | | | | | |
| `corpus.py` (how every table is counted) | | | | | |
| `categories.py` (three systems of named word classes) | | | | | |
| `README.md` | | | | | |

Program version judged: ______ (commit hash) Date: ______

Overall comments:
