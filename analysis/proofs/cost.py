"""Proof cost: the length of a proof's source code in bits (moved from the ladder's account length tool, removed
with decision 23).

Bits: the source with comments and docstrings removed (parsed and re-printed with ast.unparse, which also
normalises layout), compressed with raw LZMA2 at preset 9 | EXTREME (no container header), times 8.
"""
from __future__ import annotations

import ast
import lzma

FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def compressed_bits(data: bytes) -> int:
    return 8 * len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=FILTERS))


def strip_source(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.unparse(tree)


def source_bits(*paths: str) -> int:
    """Bits of the given source files together (stripped and concatenated in the given order)."""
    src = []
    for p in paths:
        with open(p) as f:
            src.append(strip_source(f.read()))
    return compressed_bits("\n".join(src).encode())
