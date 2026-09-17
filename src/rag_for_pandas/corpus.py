"""Extract the documentation corpus from a pandas source checkout.

pandas is parsed with `ast` rather than imported, so extraction works on plain
source files and never runs pandas code.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

# Subpackages holding the public API. Excludes _libs (Cython internals),
# tests, and compat shims: users never ask questions about those.
# _config holds set_option, get_option and option_context; util holds show_versions.
INCLUDE_SUBPACKAGES = ("core", "io", "plotting", "errors", "api", "_config", "util")

# Docstrings shorter than this are stubs like "Return the values."
# They add noise to a retrieval index without answering any real question.
MIN_DOCSTRING_WORDS = 20

# Not exported by name, but users reach their methods through public objects:
# base classes (DataFrame.to_csv lives on NDFrame) and accessors (Series.str,
# Series.dt, DataFrame.style). Taken from the Accessor(...) assignments in
# frame.py, series.py and indexes/base.py.
USER_REACHABLE_CLASSES = frozenset({
    "NDFrame", "IndexOpsMixin", "IndexingMixin",
    "GroupBy", "BaseGroupBy",
    "DatetimeIndexOpsMixin", "DatetimeTimedeltaMixin",
    # Base classes of DatetimeArray and TimedeltaArray: Series.dt.floor,
    # round, ceil and strftime are documented here.
    "DatelikeOps", "TimelikeOps",
    "StringMethods", "CategoricalAccessor", "CombinedDatetimelikeProperties",
    "DatetimeProperties", "TimedeltaProperties", "PeriodProperties",
    "SparseAccessor", "SparseFrameAccessor", "StructAccessor", "ListAccessor",
    "Styler", "StylerRenderer",
})


@dataclass
class Extraction:
    records: list[dict]  # public documentation only
    files_parsed: int
    files_failed: int
    found: int  # records before dropping internal names


def document_text(doc: dict) -> str:
    """The text every search method sees for a document.

    All methods must see identical text; otherwise a comparison measures the
    input rather than the search method.
    """
    return f"{doc['qualname']}: {doc['docstring']}"


def final_name(qualname: str) -> str:
    """'DataFrame.dropna' -> 'dropna'. Labels and search results are compared by this name."""
    return qualname.split(".")[-1]


def corpus_owners(docs: list[dict]) -> dict[str, set[str]]:
    """Map each final name to the classes documenting it: 'lower' -> {'StringMethods'}."""
    owners: dict[str, set[str]] = {}
    for doc in docs:
        owner, _, name = doc["qualname"].rpartition(".")
        owners.setdefault(name, set()).add(owner)
    return owners


def is_public(name: str) -> bool:
    """Public API names do not start with an underscore (PEP 8 convention)."""
    return not name.startswith("_")


def export_files(package: Path) -> list[Path]:
    """Modules whose __all__ lists define what pandas officially exports."""
    fixed = ["__init__.py", "arrays/__init__.py", "plotting/__init__.py", "errors/__init__.py", "core/dtypes/api.py"]
    return [package / name for name in fixed] + sorted((package / "api").rglob("__init__.py"))


def exported_names(package: Path) -> set[str]:
    """Every top-level name a user can reach from `import pandas`."""
    names = set(USER_REACHABLE_CLASSES)
    for path in export_files(package):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__":
                names.update(ast.literal_eval(node.value))
    return names


# Functions that build a property from a docstring argument, such as
# `year = _field_accessor("year", "Y", """The year of the datetime.""")`.
# The docstring is always the last positional argument.
DOCSTRING_FACTORIES = frozenset({"_field_accessor"})


def string_value(node: ast.AST, texts: dict[str, str]) -> str | None:
    """A string literal, or a name bound to one earlier at the same level.

    timedeltas.py writes `days_docstring = textwrap.dedent(\"\"\"...\"\"\")` and then
    passes `days_docstring`, so names and dedent calls are resolved too.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return texts.get(node.id)
    if isinstance(node, ast.Call) and len(node.args) == 1:
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        if name == "dedent":
            return string_value(node.args[0], texts)
    return None


def doc_argument(node: ast.AST, texts: dict[str, str]) -> str | None:
    """Docstring passed into a call instead of written under a `def`.

    Two forms: `columns = AxisProperty(axis=0, doc=...)` and the property
    factories in DOCSTRING_FACTORIES. These attributes are assignments, not
    `def` or `class`, so ast.get_docstring cannot see them.
    """
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
        return None
    call = node.value
    for keyword in call.keywords:
        if keyword.arg == "doc":
            text = string_value(keyword.value, texts)
            return inspect.cleandoc(text) if text else None
    if getattr(call.func, "id", "") in DOCSTRING_FACTORIES and call.args:
        text = string_value(call.args[-1], texts)
        return inspect.cleandoc(text) if text else None
    return None


def text_assignments(node: ast.AST) -> dict[str, str]:
    """Names assigned a string at this level: `days_docstring = dedent(\"\"\"...\"\"\")`."""
    texts: dict[str, str] = {}
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
            text = string_value(child.value, texts)
            if text is not None:
                texts[child.targets[0].id] = text
    return texts


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
    texts = text_assignments(node)
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
                record(doc_argument(child, texts), prefix + target.id, "attribute", source_file, out)


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


def extract(package: Path) -> Extraction:
    """Parse the included subpackages and keep documentation for user-reachable names."""
    records: list[dict] = []
    files_parsed = files_failed = 0
    for subpackage in INCLUDE_SUBPACKAGES:
        for path in (package / subpackage).rglob("*.py"):
            if "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                files_failed += 1
                continue
            files_parsed += 1
            collect(tree, "", path, records)

    exported = exported_names(package)
    public = [r for r in records if r["qualname"].split(".")[0] in exported]
    return Extraction(public, files_parsed, files_failed, len(records))
