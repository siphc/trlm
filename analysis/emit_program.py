"""Write a fitted Student (analysis/student.py) into program/tables/ as readable Python tables.

Writes program/tables/follow.py and program/tables/head<h>.py. Values are rounded to DECIMALS places.
Shift-invariant quantities are centred for readability (a constant added to every score in a row does
not change the softmax): WORD and SALIENCE by their median, rows of CLASS_FOLLOW and PREFER by the
row median. Entries that round to zero are omitted (a missing entry means 0).

Each table file names the class systems of program/categories.py that its class tables use: the system whose
token map equals the student's map (student.cls -> PREFER; student.ctx -> rows of CLASS_FOLLOW and CLASS_RECALL,
and COPY; student.tgt -> their columns). Per-token tables are grouped by the matching system (WORD by the
column system, SALIENCE by the attention system) and exceptions by the column system's class of the next token.

Usage (library): emit(student, model_id, header_note)
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.gen_categories import ID, TOKENS, class_maps, class_systems  # noqa: E402

CLASS_SYSTEMS = class_systems()
TOKEN_CLASS = class_maps()

DECIMALS = 2
MAX_LITERAL = 64  # harness lint: numbers per literal


def _fmt(x: float) -> str:
    s = f"{x:.{DECIMALS}f}"
    return "0.0" if s in ("-0.00", "0.00") else s


def _dict_lines(d: dict, indent: str, width: int = 96) -> list[str]:
    items = [f"{k!r}: {_fmt(v)}" for k, v in d.items()]
    lines, row = [], []
    for it in items:
        if row and len(indent) + sum(len(r) + 2 for r in row) + len(it) > width:
            lines.append(indent + ", ".join(row) + ",")
            row = []
        row.append(it)
    if row:
        lines.append(indent + ", ".join(row) + ",")
    return lines


def _nonzero(d: dict) -> dict:
    return {k: v for k, v in d.items() if _fmt(v) != "0.0"}


def system_of(class_map: np.ndarray) -> str:
    """Name of the class system in program/categories.py whose token -> class map is class_map."""
    for name in CLASS_SYSTEMS:
        if np.array_equal(TOKEN_CLASS[name], np.asarray(class_map)):
            return name
    raise ValueError("the student's class map matches no class system in program/categories.py")


def per_token_block(name: str, values: np.ndarray, doc: str, system: str) -> list[str]:
    out = [f"# {doc}", f"{name} = {{}}"]
    for cname, toks in CLASS_SYSTEMS[system].items():
        d = _nonzero({t: float(values[ID[t]]) for t in toks})
        if not d:
            continue
        out.append(f"{name}[{cname!r}] = {{")
        out += _dict_lines(d, "    ")
        out.append("}")
    return out + [""]


def class_matrix_block(name: str, M: np.ndarray, doc: str, rows: str, columns: str) -> list[str]:
    out = [f"# {doc}", f"{name} = {{}}"]
    col_names = list(CLASS_SYSTEMS[columns])
    for a, an in enumerate(CLASS_SYSTEMS[rows]):
        d = _nonzero({bn: float(M[a, b]) for b, bn in enumerate(col_names)})
        if not d:
            continue
        out.append(f"{name}[{an!r}] = {{")
        out += _dict_lines(d, "    ")
        out.append("}")
    return out + [""]


def per_class_block(name: str, vals: np.ndarray, doc: str, system: str) -> list[str]:
    """One literal per MAX_LITERAL classes: NAME = {...}, then NAME.update({...}) for the rest."""
    items = list(_nonzero({cn: float(vals[k]) for k, cn in enumerate(CLASS_SYSTEMS[system])}).items())
    out = [f"# {doc}"]
    for i in range(0, max(len(items), 1), MAX_LITERAL):
        chunk = dict(items[i:i + MAX_LITERAL])
        out += [f"{name} = {{" if i == 0 else f"{name}.update({{"] + _dict_lines(chunk, "    ")
        out.append("}" if i == 0 else "})")
    return out + [""]


def pairs_block(name: str, M: np.ndarray, doc: str, columns: str) -> list[str]:
    out = [f"# {doc}", f"{name} = {{}}", ""]
    col_names = list(CLASS_SYSTEMS[columns])
    class_of = TOKEN_CLASS[columns]
    rows = np.nonzero(np.abs(M).max(1) >= 0.5 * 10 ** -DECIMALS)[0]
    for t in rows:
        cols = np.nonzero(np.abs(M[t]) >= 0.5 * 10 ** -DECIMALS)[0]
        groups: dict[str, dict] = {}
        for v in sorted(cols, key=lambda v: -abs(M[t, v])):
            groups.setdefault(col_names[class_of[v]], {})[TOKENS[v]] = float(M[t, v])
        groups = {k: _nonzero(g) for k, g in groups.items() if _nonzero(g)}
        if not groups:
            continue
        out.append(f"pair({name}, {TOKENS[t]!r},")
        for cn, g in groups.items():
            out.append(f"     {cn}={{")
            out += _dict_lines(g, "         ")
            out.append("     },")
        out.append(")")
    return out + [""]


def emit(st, model_id: str, note: str) -> dict:
    sd = {k: v.detach().cpu().double().numpy() for k, v in st.state_dict().items()}
    att = system_of(sd["cls"].astype(int))
    rows = system_of(sd["ctx"].astype(int) if "ctx" in sd else sd["cls"].astype(int))
    columns = system_of(sd["tgt"].astype(int) if "tgt" in sd else sd["cls"].astype(int))
    tdir = os.path.join(REPO, "program", "tables")
    os.makedirs(tdir, exist_ok=True)
    head = [f'"""Generated by analysis/emit_program.py. {note}', "",
            f"Fitted by regression to the log-probabilities of model {model_id} on TRAIN windows",
            "(minimising KL(model || program)); see analysis/student.py for the fitting procedure.",
            'All values are in nats and are added to next-token scores before the softmax."""', ""]

    # FOLLOW
    word = sd["word"] - np.median(sd["word"])
    cf = sd["class_follow"] - np.median(sd["class_follow"], axis=1, keepdims=True)
    exc = sd["follow_exc"] * sd["follow_mask"]
    lines = head + ["from ..tables_lib import pair", ""]
    lines += ["# Class systems of CLASS_FOLLOW (program/categories.py): rows = class of the current token,",
              "# columns = class of the next token.", f"ROWS = {rows!r}", f"COLUMNS = {columns!r}", ""]
    lines += per_token_block("WORD", word, "How likely each token is as a next token in general (higher = more common).",
                             columns)
    lines += class_matrix_block("CLASS_FOLLOW", cf, f"Row = {rows} class of the current token, column = {columns} "
                                "class of the next token.", rows, columns)
    lines += pairs_block("AFTER", exc, "Exceptions: pair(AFTER, current token, CLASS_OF_NEXT={next token: extra score}).",
                         columns)
    with open(os.path.join(tdir, "follow.py"), "w") as f:
        f.write("\n".join(lines))

    from analysis.student import BUCKET_START

    H = sd["salience"].shape[0]
    for h in range(H):
        sal = sd["salience"][h] - np.median(sd["salience"][h])
        pref = sd["prefer"][h] - np.median(sd["prefer"][h], axis=1, keepdims=True)
        rec = sd["recency"][h] - sd["recency"][h].max()
        lines = head + ["from ..tables_lib import pair", ""]
        sv = bool(float(sd.get("self_votes", np.array(1.0))) != 0.0)
        lines += ["# False: a head looking at the current token casts no vote there (see attention.py).",
                  f"SELF_VOTES = {sv}", ""]
        lines += ["# Class systems (program/categories.py): PREFER compares the classes of the current and the attended",
                  "# token in PREFER_CLASSES; CLASS_RECALL rows (and COPY) use the attended token's class in RECALL_ROWS,",
                  "# its columns the next token's class in RECALL_COLUMNS.",
                  f"PREFER_CLASSES = {att!r}", f"RECALL_ROWS = {rows!r}", f"RECALL_COLUMNS = {columns!r}", ""]
        lines += [f"# Distance buckets: bucket b covers distances [RECENCY_BUCKET_START[b], RECENCY_BUCKET_START[b + 1]).",
                  f"RECENCY_BUCKET_START = {BUCKET_START!r}",
                  "# How much the head prefers a position at that distance (0 = most preferred).",
                  "RECENCY = [" + ", ".join(_fmt(x) for x in rec) + "]", ""]
        lines += per_token_block("SALIENCE", sal, "How strongly each token attracts this head's attention.", att)
        lines += class_matrix_block("PREFER", pref, f"Row = {att} class of the current token, column = {att} class "
                                    "of the attended token.", att, att)
        lines += class_matrix_block("CLASS_RECALL", sd["class_recall"][h], f"Row = {rows} class of the attended "
                                    f"token, column = {columns} class of the voted next token.", rows, columns)
        lines += per_class_block("COPY", sd["copy"][h], f"Extra vote for repeating the attended token itself, by its "
                                 f"{rows} class.", rows)
        lines += pairs_block("RECALL", sd["recall_exc"][h] * sd["recall_mask"][h],
                             "Exceptions: pair(RECALL, attended token, CLASS_OF_NEXT={next token: extra vote}).", columns)
        with open(os.path.join(tdir, f"head{h}.py"), "w") as f:
            f.write("\n".join(lines))
    conf = sd.get("log_conf", np.zeros(len(TOKENS)))
    if np.abs(conf).max() > 0:
        raise NotImplementedError("CONFIDENCE was fitted: re-add program/confidence.py (removed as a no-op) first")

    for h in range(H, 4):  # heads the student does not model add nothing
        with open(os.path.join(tdir, f"head{h}.py"), "w") as f:
            f.write(EMPTY_HEAD)
    return {"follow_exceptions": int((np.abs(exc) >= 0.005).sum()), "class_systems": {
        "prefer": att, "rows": rows, "columns": columns}}


EMPTY_HEAD = '''"""Empty tables: this head adds nothing in this version of the program."""
SELF_VOTES = True
PREFER_CLASSES = "ATTENTION"
RECALL_ROWS = "ATTENTION"
RECALL_COLUMNS = "ATTENTION"
RECENCY_BUCKET_START = [0]
RECENCY = [0.0]
SALIENCE = {}
PREFER = {}
CLASS_RECALL = {}
COPY = {}
RECALL = {}
'''
