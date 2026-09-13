"""Measure the API documentation embedded in pandas source code.

Exploration script for corpus design: answers how much docstring text exists
and whether it is usable, before committing to an ingestion pipeline.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

PANDAS_PACKAGE = Path("data/raw/pandas/pandas")
OUTPUT_PATH = Path("data/interim/docstrings.jsonl")

# Subpackages holding the public API. Excludes _libs (Cython internals),
# tests, and compat shims: users never ask questions about those.
INCLUDE_SUBPACKAGES = ("core", "io", "plotting", "errors", "api")

# Docstrings shorter than this are stubs like "Return the values."
# They add noise to a retrieval index without answering any real question.
MIN_DOCSTRING_WORDS = 20


def is_public(name: str) -> bool:
    """Public API names do not start with an underscore (PEP 8 convention)."""
    return not name.startswith("_")


def collect(node: ast.AST, prefix: str, source_file: Path, out: list[dict]) -> None:
    """Walk one level of the tree, recording docstrings and recursing into classes.

    `prefix` carries the enclosing class name so a method is recorded as
    "DataFrame.merge" rather than a bare "merge", which would be ambiguous
    across the dozens of classes that define a method of the same name.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            if not is_public(child.name):
                continue
            qualname = prefix + child.name
            record(child, qualname, "class", source_file, out)
            collect(child, qualname + ".", source_file, out)

        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(child.name):
                continue
            record(child, prefix + child.name, "function", source_file, out)


def record(node: ast.AST, qualname: str, kind: str, source_file: Path, out: list[dict]) -> None:
    """Append one docstring record if it passes the quality filters."""
    docstring = ast.get_docstring(node, clean=True)
    if not docstring:
        return

    word_count = len(docstring.split())
    if word_count < MIN_DOCSTRING_WORDS:
        return

    out.append(
        {
            "qualname": qualname,
            "kind": kind,
            "source_file": str(source_file).replace("\\", "/"),
            "word_count": word_count,
            "docstring": docstring,
        }
    )


def main() -> None:
    records: list[dict] = []
    files_parsed = 0
    files_failed = 0

    for subpackage in INCLUDE_SUBPACKAGES:
        for path in (PANDAS_PACKAGE / subpackage).rglob("*.py"):
            if "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                files_failed += 1
                continue
            files_parsed += 1
            collect(tree, "", path, records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record_ in records:
            handle.write(json.dumps(record_) + "\n")

    total_words = sum(r["word_count"] for r in records)
    print(f"files parsed:     {files_parsed}")
    print(f"files failed:     {files_failed}")
    print(f"docstrings kept:  {len(records)}")
    print(f"total words:      {total_words:,}")
    print(f"median words:     {sorted(r['word_count'] for r in records)[len(records) // 2]}")
    print(f"written to:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
