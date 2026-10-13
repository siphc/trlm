# Human judging sheet (AGENTS.md section 12)

Read `program/` top-down, blind to the metrics if possible: start with `program/README.md`, then
`program/__init__.py`, `follow.py`, `attention.py`, `head0.py` … `head3.py`, and skim `tables/` and
`categories.py`. For each component answer the four questions with 0 (no), 1 (partly) or 2 (yes), and write
concrete rewrite requests for anything scored below 2. The agent turns each request into a change.

Questions:
1. Can I state in one sentence what it does?
2. Can I predict what happens to the output if it is removed?
3. Do names and comments match the behaviour?
4. Is any table or constant opaque to me? (2 = nothing opaque)

| component | Q1 one sentence | Q2 predict removal | Q3 names match | Q4 nothing opaque | rewrite requests |
|---|---|---|---|---|---|
| `__init__.py` (the algorithm) | | | | | |
| `follow.py` + `tables/follow.py` | | | | | |
| `attention.py` (abstention rule) | | | | | |
| `head0.py` + `tables/head0.py` | | | | | |
| `head1.py` + `tables/head1.py` | | | | | |
| `head2.py` + `tables/head2.py` | | | | | |
| `head3.py` + `tables/head3.py` | | | | | |
| `categories.py` (three class systems) | | | | | |
| `README.md` | | | | | |

Program version judged: ______ (commit hash) Date: ______

Overall comments:
