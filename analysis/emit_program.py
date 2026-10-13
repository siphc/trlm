"""Write a fitted Student (analysis/student.py) into program/tables/ as readable Python tables.

Writes program/tables/follow.py and program/tables/head<h>.py. Values are rounded to DECIMALS places.
Shift-invariant quantities are centred for readability (a constant added to every score in a row does
not change the softmax): WORD and SALIENCE by their median, rows of CLASS_FOLLOW and PREFER by the
row median. Entries that round to zero are omitted (a missing entry means 0).

Usage (library): emit(student, model_id, header_note)
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from program.categories import CLASS_NAMES, CLASSES  # noqa: E402
from program.vocab import ID, TOKENS  # noqa: E402

DECIMALS = 2
TOKEN_CLASS = None


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


def per_token_block(name: str, values: np.ndarray, doc: str) -> list[str]:
    out = [f"# {doc}", f"{name} = {{}}"]
    for cname in CLASS_NAMES:
        d = _nonzero({t: float(values[ID[t]]) for t in CLASSES[cname]})
        if not d:
            continue
        out.append(f"{name}[{cname!r}] = {{")
        out += _dict_lines(d, "    ")
        out.append("}")
    return out + [""]


def class_matrix_block(name: str, M: np.ndarray, doc: str) -> list[str]:
    out = [f"# {doc}", f"{name} = {{}}"]
    for a, an in enumerate(CLASS_NAMES):
        d = _nonzero({bn: float(M[a, b]) for b, bn in enumerate(CLASS_NAMES)})
        if not d:
            continue
        out.append(f"{name}[{an!r}] = {{")
        out += _dict_lines(d, "    ")
        out.append("}")
    return out + [""]


def per_class_block(name: str, vals: np.ndarray, doc: str) -> list[str]:
    d = _nonzero({cn: float(vals[k]) for k, cn in enumerate(CLASS_NAMES)})
    return [f"# {doc}", f"{name} = {{"] + _dict_lines(d, "    ") + ["}", ""]


def pairs_block(name: str, M: np.ndarray, doc: str, class_of: np.ndarray, imp: np.ndarray | None = None,
                row_note=None) -> list[str]:
    """Token-pair exceptions. With `imp` (same shape as M): rows in decreasing total importance, and inside a
    row boosts first, then suppressions, each in decreasing importance. `row_note(t)` adds a comment line."""
    out = [f"# {doc}", f"{name} = {{}}", ""]
    rows = np.nonzero(np.abs(M).max(1) >= 0.5 * 10 ** -DECIMALS)[0]
    if imp is not None:
        rows = sorted(rows, key=lambda t: -float(imp[t].sum()))
    for t in rows:
        cols = np.nonzero(np.abs(M[t]) >= 0.5 * 10 ** -DECIMALS)[0]
        if imp is None:
            order = sorted(cols, key=lambda v: -abs(M[t, v]))
        else:
            order = (sorted([v for v in cols if M[t, v] > 0], key=lambda v: -imp[t, v])
                     + sorted([v for v in cols if M[t, v] < 0], key=lambda v: -imp[t, v]))
        groups: dict[str, dict] = {}
        for v in order:
            groups.setdefault(CLASS_NAMES[class_of[v]], {})[TOKENS[v]] = float(M[t, v])
        groups = {k: _nonzero(g) for k, g in groups.items() if _nonzero(g)}
        if not groups:
            continue
        if row_note is not None:
            out.append(row_note(t))
        out.append(f"pair({name}, {TOKENS[t]!r},")
        for cn, g in groups.items():
            out.append(f"     {cn}={{")
            out += _dict_lines(g, "         ")
            out.append("     },")
        out.append(")")
    return out + [""]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def emit(st, model_id: str, note: str, out_dir: str | None = None, decimals: int | None = None,
         readable: bool = False, windows=None) -> dict:
    """Write tables. readable=True (phase-6 format): rows and entries ordered by second-order importance
    (student.fisher_importance on `windows`), boosts before suppressions, each FOLLOW row annotated with the
    top probabilities FOLLOW alone gives, head tables annotated with typical attention weights."""
    import torch

    global DECIMALS
    if decimals is not None:
        DECIMALS = decimals
    sd = {k: v.detach().cpu().double().numpy() for k, v in st.state_dict().items()}
    class_of = sd["cls"].astype(int)
    imp_f = imp_r = None
    att_stats = None
    if readable:
        from analysis.student import fisher_importance

        st_cpu = st.cpu()
        imp_f, imp_r = (x.double().numpy() for x in fisher_importance(st_cpu, windows, n_batches=40, bs=8))
        att_stats = _attention_stats(st_cpu, windows[:64])
    tdir = os.path.join(out_dir or os.path.join(REPO, "program"), "tables")
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
    lines += per_token_block("WORD", word, "How likely each token is as a next token in general (higher = more common).")
    lines += class_matrix_block("CLASS_FOLLOW", cf, "Row = class of the current token, column = class of the next token.")
    follow_full = word[None, :] + cf[class_of][:, class_of] + exc if readable else None

    def follow_note(t):
        q = _softmax(follow_full[t])
        top = np.argsort(-q)[:3]
        return "# after " + repr(TOKENS[t]) + " (FOLLOW alone): " + ", ".join(
            f"{TOKENS[v]!r} {100 * q[v]:.0f}%" for v in top)

    lines += pairs_block("AFTER", exc, "Exceptions: pair(AFTER, current token, CLASS_OF_NEXT={next token: extra score})."
                         + (" Rows in order of importance; boosts first." if readable else ""), class_of,
                         imp=imp_f, row_note=follow_note if readable else None)
    with open(os.path.join(tdir, "follow.py"), "w") as f:
        f.write("\n".join(lines))

    from analysis.student import BUCKET_START

    H = sd["salience"].shape[0]
    for h in range(H):
        sal = sd["salience"][h] - np.median(sd["salience"][h])
        pref = sd["prefer"][h] - np.median(sd["prefer"][h], axis=1, keepdims=True)
        rec = sd["recency"][h] - sd["recency"][h].max()
        lines = head + ["from ..tables_lib import pair", ""]
        if att_stats is not None:
            a = att_stats[h]
            lines += [f"# Reading the votes: this head puts {100 * a['self']:.0f}% of its attention on the current token "
                      "(where it abstains) and the rest on earlier tokens.",
                      f"# A token it attends to typically carries weight {a['typ']:.2f} (attention-weighted mean), so a "
                      f"RECALL or COPY value of 10 nats adds about {10 * a['typ']:.1f} nats there.", ""]
        sv = bool(float(sd.get("self_votes", np.array(1.0))) != 0.0)
        lines += ["# False: a head looking at the current token casts no vote there (see attention.py).",
                  f"SELF_VOTES = {sv}", ""]
        lines += [f"# Distance buckets: bucket b covers distances [RECENCY_BUCKET_START[b], RECENCY_BUCKET_START[b + 1]).",
                  f"RECENCY_BUCKET_START = {BUCKET_START!r}",
                  "# How much the head prefers a position at that distance (0 = most preferred).",
                  "RECENCY = [" + ", ".join(_fmt(x) for x in rec) + "]", ""]
        lines += per_token_block("SALIENCE", sal, "How strongly each token attracts this head's attention.")
        lines += class_matrix_block("PREFER", pref, "Row = class of the current token, column = class of the attended token.")
        lines += class_matrix_block("CLASS_RECALL", sd["class_recall"][h],
                                    "Row = class of the attended token, column = class of the voted next token.")
        lines += per_class_block("COPY", sd["copy"][h], "Extra vote for repeating the attended token itself, by its class.")
        lines += pairs_block("RECALL", sd["recall_exc"][h] * sd["recall_mask"][h],
                             "Exceptions: pair(RECALL, attended token, CLASS_OF_NEXT={next token: extra vote})."
                             + (" Rows in order of importance; boosts first." if readable else ""), class_of,
                             imp=None if imp_r is None else imp_r[h])
        with open(os.path.join(tdir, f"head{h}.py"), "w") as f:
            f.write("\n".join(lines))
    conf = sd.get("log_conf", np.zeros(len(TOKENS)))
    if np.abs(conf).max() > 0:
        raise NotImplementedError("CONFIDENCE was fitted: re-add program/confidence.py (removed as a no-op) first")

    for h in range(H, 4):  # heads the student does not model add nothing
        with open(os.path.join(tdir, f"head{h}.py"), "w") as f:
            f.write(EMPTY_HEAD)
    return {"follow_exceptions": int((np.abs(exc) >= 0.005).sum())}


def _attention_stats(st, W):
    """Per head: mean attention on the current token, and the attention-weighted mean weight on earlier tokens."""
    import torch

    out = []
    with torch.no_grad():
        a = st.attention(torch.as_tensor(np.asarray(W)[:, :256]).long())          # (B, H, T, T)
    for h in range(a.shape[1]):
        ah = a[:, h].double()
        diag = torch.diagonal(ah, dim1=-2, dim2=-1)
        off = ah - torch.diag_embed(diag)
        mass = off.sum(-1).clamp(min=1e-12)
        typ = ((off ** 2).sum(-1) / mass)
        out.append({"self": float(diag.mean()), "typ": float((typ * mass).sum() / mass.sum())})
    return out


EMPTY_HEAD = '''"""Empty tables: this head adds nothing in this version of the program."""
SELF_VOTES = True
RECENCY_BUCKET_START = [0]
RECENCY = [0.0]
SALIENCE = {}
PREFER = {}
CLASS_RECALL = {}
COPY = {}
RECALL = {}
'''
