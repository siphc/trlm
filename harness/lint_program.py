"""Static lint for the replacement program (AGENTS.md section 8). READ-ONLY for replacement agents.

Usage: python -m harness.lint_program [program_dir]   (default: program)

Checks every .py file under the program directory except tests/:
  - imports: Python stdlib allowlist + numpy only; relative imports inside the package are fine
  - forbidden calls/names: open, eval, exec, compile, __import__, input, breakpoint, globals, vars,
    np.load / np.loadtxt / np.genfromtxt / np.fromfile / np.frombuffer / np.memmap / np.fromstring
  - numeric literals: any list/tuple/set/dict literal holding more than MAX_LITERAL numeric elements
    (counted over the whole nested literal) fails
  - string literals: non-docstring strings longer than MAX_STRING characters fail; any string
    (docstrings included) containing a run of more than MAX_TOKEN_RUN non-whitespace characters fails
    (catches base64/hex payloads); bytes literals longer than 64 fail
Reports counts: lines of code, numeric literals, largest literal.
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import sys
import tokenize

MAX_LITERAL = 64          # configs/frozen.yaml: lint_max_numeric_elements_per_literal
MAX_STRING = 400          # longest allowed non-docstring string literal
MAX_TOKEN_RUN = 80        # longest allowed run of non-whitespace characters in any string
MAX_BYTES = 64

ALLOWED_MODULES = {
    "numpy", "math", "cmath", "functools", "itertools", "collections", "typing", "dataclasses",
    "enum", "string", "numbers", "abc", "operator", "statistics", "bisect", "heapq", "__future__",
    "re", "fractions", "decimal", "unicodedata", "textwrap", "warnings", "contextlib", "copy",
    "types", "random",
}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
                   "globals", "vars", "memoryview", "__builtins__", "getattr", "setattr"}
FORBIDDEN_NP_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "frombuffer", "memmap",
                      "fromstring", "save", "savez", "savetxt", "DataSource", "lib", "ctypeslib"}


def _is_numeric_const(n: ast.AST) -> bool:
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        n = n.operand
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)) \
        and not isinstance(n.value, bool)


def _literal_numeric_count(n: ast.AST) -> int:
    """Numeric elements inside a (possibly nested) container literal."""
    if _is_numeric_const(n):
        return 1
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        return sum(_literal_numeric_count(e) for e in n.elts)
    if isinstance(n, ast.Dict):
        return sum(_literal_numeric_count(e) for e in list(n.keys) + list(n.values) if e is not None)
    return 0


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _loc(src: str) -> int:
    """Non-blank lines that are not pure comments (docstrings count as code lines)."""
    n = 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return sum(1 for line in src.splitlines() if line.strip())
    lines = set()
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                      tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        for ln in range(t.start[0], t.end[0] + 1):
            lines.add(ln)
    n = len(lines)
    return n


def lint_file(path: str, rel: str) -> dict:
    src = open(path, encoding="utf-8").read()
    errors: list[str] = []
    tree = ast.parse(src, filename=path)
    docstrings = _docstring_nodes(tree)
    n_numeric = 0
    largest = 0
    counted_containers: set[int] = set()

    for node in ast.walk(tree):
        loc = f"{rel}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    errors.append(f"{loc}: import of '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_MODULES:
                    errors.append(f"{loc}: import from '{node.module}' not allowed")
                if root == "numpy":
                    for a in node.names:
                        if a.name in FORBIDDEN_NP_ATTRS:
                            errors.append(f"{loc}: numpy.{a.name} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"{loc}: use of '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NP_ATTRS and isinstance(node.value, ast.Name) \
                    and node.value.id in ("np", "numpy"):
                errors.append(f"{loc}: np.{node.attr} not allowed")
            if node.attr.startswith("__") and node.attr not in ("__init__", "__name__", "__doc__"):
                errors.append(f"{loc}: dunder attribute access '{node.attr}' not allowed")
        elif isinstance(node, ast.Constant):
            if _is_numeric_const(node):
                n_numeric += 1
            elif isinstance(node.value, str):
                s = node.value
                if id(node) not in docstrings and len(s) > MAX_STRING:
                    errors.append(f"{loc}: string literal of {len(s)} chars (> {MAX_STRING})")
                run = max((len(m) for m in re.findall(r"\S+", s)), default=0)
                if run > MAX_TOKEN_RUN:
                    errors.append(f"{loc}: string contains a {run}-char unbroken run (> {MAX_TOKEN_RUN})")
            elif isinstance(node.value, bytes) and len(node.value) > MAX_BYTES:
                errors.append(f"{loc}: bytes literal of {len(node.value)} bytes (> {MAX_BYTES})")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)) and id(node) not in counted_containers:
            c = _literal_numeric_count(node)
            for sub in ast.walk(node):
                counted_containers.add(id(sub))
            largest = max(largest, c)
            if c > MAX_LITERAL:
                errors.append(f"{loc}: literal with {c} numeric elements (> {MAX_LITERAL})")

    return {"file": rel, "loc": _loc(src), "numeric_literals": n_numeric,
            "largest_literal": largest, "errors": errors}


def lint_program(program_dir: str = "program") -> dict:
    files = []
    for root, dirs, fnames in os.walk(program_dir):
        rel_root = os.path.relpath(root, program_dir)
        if rel_root.split(os.sep)[0] in ("tests", "__pycache__"):
            continue
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
        for fn in sorted(fnames):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                files.append(lint_file(p, os.path.relpath(p, program_dir)))
    errors = [e for f in files for e in f["errors"]]
    return {
        "ok": not errors,
        "loc": sum(f["loc"] for f in files),
        "numeric_literals": sum(f["numeric_literals"] for f in files),
        "largest_literal": max((f["largest_literal"] for f in files), default=0),
        "n_files": len(files),
        "errors": errors,
    }


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "program"
    r = lint_program(d)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)
