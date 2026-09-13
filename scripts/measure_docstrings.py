"""Measure the API documentation embedded in pandas source code.

Exploration script for corpus design: answers how much docstring text exists
and whether it is usable, before committing to an ingestion pipeline.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

PANDAS_PACKAGE = Path("data/raw/pandas/pandas")
OUTPUT_PATH = Path("data/interim/docstrings.jsonl")

# Subpackages holding the public API. Excludes _libs (Cython internals),
# tests, and compat shims: users never ask questions about those.
# _config holds set_option, get_option and option_context.
INCLUDE_SUBPACKAGES = ("core", "io", "plotting", "errors", "api", "_config")

# Docstrings shorter than this are stubs like "Return the values."
# They add noise to a retrieval index without answering any real question.
MIN_DOCSTRING_WORDS = 20

# Modules whose __all__ lists define what pandas officially exports.
EXPORT_FILES = (
    "__init__.py",
    "arrays/__init__.py",
    "plotting/__init__.py",
    "errors/__init__.py",
    "core/dtypes/api.py",
    *(str(p.relative_to(PANDAS_PACKAGE)) for p in sorted((PANDAS_PACKAGE / "api").rglob("__init__.py"))),
)

# Not exported by name, but users reach their methods through public objects:
# base classes (DataFrame.to_csv lives on NDFrame) and accessors (Series.str,
# Series.dt, DataFrame.style). Taken from the Accessor(...) assignments in
# frame.py, series.py and indexes/base.py.
USER_REACHABLE_CLASSES = {
    "NDFrame", "IndexOpsMixin", "IndexingMixin",
    "GroupBy", "BaseGroupBy",
    "DatetimeIndexOpsMixin", "DatetimeTimedeltaMixin",
    "StringMethods", "CategoricalAccessor", "CombinedDatetimelikeProperties",
    "DatetimeProperties", "TimedeltaProperties", "PeriodProperties",
    "SparseAccessor", "SparseFrameAccessor", "StructAccessor", "ListAccessor",
    "Styler", "StylerRenderer",
}


def is_public(name: str) -> bool:
    """Public API names do not start with an underscore (PEP 8 convention)."""
    return not name.startswith("_")


def exported_names() -> set[str]:
    """Every top-level name a user can reach from `import pandas`."""
    names = set(USER_REACHABLE_CLASSES)
    for relative in EXPORT_FILES:
        tree = ast.parse((PANDAS_PACKAGE / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__":
                names.update(ast.literal_eval(node.value))
    return names


def doc_keyword(node: ast.AST) -> str | None:
    """Docstring passed as `doc="..."`, as in `columns = AxisProperty(axis=0, doc=...)`.

    These attributes are assignments, not `def` or `class`, so
    ast.get_docstring cannot see them.
    """
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
        return None
    for keyword in node.value.keywords:
        if keyword.arg == "doc" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return inspect.cleandoc(keyword.value.value)
    return None


def collect(node: ast.AST, prefix: str, source_file: Path, out: list[dict]) -> None:
    """Walk one level of the tree, recording docstrings and recursing into classes.

    `prefix` carries the enclosing class name so a method is recorded as
    "DataFrame.merge" rather than a bare "merge", which would be ambiguous
    across the dozens of classes that define a method of the same name.
    """
    # Docstrings of this level's functions, so an alias such as
    # `agg = aggregate` can reuse the documentation of the method it points to.
    method_docs = {
        child.name: ast.get_docstring(child)
        for child in ast.iter_child_nodes(node)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            if not is_public(child.name):
                continue
            qualname = prefix + child.name
            record(ast.get_docstring(child), qualname, "class", source_file, out)
            collect(child, qualname + ".", source_file, out)

        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(child.name):
                continue
            record(ast.get_docstring(child), prefix + child.name, "function", source_file, out)

        elif prefix and len(getattr(child, "targets", ())) == 1:
            target = child.targets[0]
            if not (isinstance(target, ast.Name) and is_public(target.id)):
                continue
            if isinstance(child.value, ast.Name) and child.value.id in method_docs:
                record(method_docs[child.value.id], prefix + target.id, "alias", source_file, out)
            else:
                record(doc_keyword(child), prefix + target.id, "attribute", source_file, out)


def record(docstring: str | None, qualname: str, kind: str, source_file: Path, out: list[dict]) -> None:
    """Append one docstring record if it passes the quality filters."""
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

    exported = exported_names()
    extracted = len(records)
    records = [r for r in records if r["qualname"].split(".")[0] in exported]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record_ in records:
            handle.write(json.dumps(record_) + "\n")

    total_words = sum(r["word_count"] for r in records)
    print(f"files parsed:     {files_parsed}")
    print(f"files failed:     {files_failed}")
    print(f"docstrings found: {extracted}")
    print(f"dropped internal: {extracted - len(records)}")
    print(f"docstrings kept:  {len(records)}")
    print(f"total words:      {total_words:,}")
    print(f"median words:     {sorted(r['word_count'] for r in records)[len(records) // 2]}")
    print(f"written to:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
