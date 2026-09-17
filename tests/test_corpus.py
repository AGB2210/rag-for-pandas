import ast
from pathlib import Path

from rag_for_pandas.corpus import INCLUDE_SUBPACKAGES, collect, corpus_owners, exported_names, extract

LONG = " ".join(f"word{i}" for i in range(25))

SOURCE = f'''
class Frame:
    """{LONG}"""

    def aggregate(self):
        """{LONG}"""

    agg = aggregate

    def short(self):
        """Too short to keep."""

    def _private(self):
        """{LONG}"""

    columns = AxisProperty(axis=0, doc="""{LONG}""")

    year = _field_accessor("year", "Y", """{LONG}""")

    days_docstring = textwrap.dedent("""{LONG}""")
    days = _field_accessor("days", "days", days_docstring)

    unknown = _field_accessor("unknown", missing_name)

    constant = compute("""{LONG}""")


def read_thing():
    """{LONG}"""


class _Hidden:
    """{LONG}"""
'''


def test_collect_records_classes_methods_aliases_and_docstring_arguments():
    out: list[dict] = []
    collect(ast.parse(SOURCE), "", Path("pkg/module.py"), out)

    # The docstring variable itself and calls to other functions are not documents.
    assert [(r["qualname"], r["kind"]) for r in out] == [
        ("Frame", "class"),
        ("Frame.aggregate", "function"),
        ("Frame.agg", "alias"),
        ("Frame.columns", "attribute"),
        ("Frame.year", "attribute"),
        ("Frame.days", "attribute"),
        ("read_thing", "function"),
    ]
    assert all(r["docstring"] == LONG and r["word_count"] == 25 for r in out)


def test_corpus_owners_groups_classes_by_final_name():
    docs = [{"qualname": "DataFrame.sum"}, {"qualname": "Series.sum"}, {"qualname": "read_csv"}]
    assert corpus_owners(docs) == {"sum": {"DataFrame", "Series"}, "read_csv": {""}}


def make_package(root: Path) -> Path:
    package = root / "pandas"
    for subpackage in INCLUDE_SUBPACKAGES:
        (package / subpackage).mkdir(parents=True)
    files = {
        "__init__.py": '__all__ = ["DataFrame", "read_csv"]',
        "arrays/__init__.py": "__all__ = []",
        "plotting/__init__.py": "__all__ = []",
        "errors/__init__.py": "__all__ = []",
        "core/dtypes/api.py": '__all__ = ["is_bool"]',
        "api/__init__.py": '__all__ = ["types"]',
        "api/types/__init__.py": '__all__ = ["infer_dtype"]',
        "core/frame.py": f'class DataFrame:\n    """{LONG}"""\n\nclass Block:\n    """{LONG}"""\n',
        "core/tests/test_frame.py": f'class DataFrame:\n    """{LONG}"""\n',
    }
    for relative, text in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return package


def test_exported_names_reads_all_lists_and_reachable_classes(tmp_path):
    names = exported_names(make_package(tmp_path))
    assert {"DataFrame", "read_csv", "is_bool", "types", "infer_dtype", "NDFrame"} <= names
    assert "Block" not in names


def test_extract_skips_tests_and_drops_internal_names(tmp_path):
    result = extract(make_package(tmp_path))
    assert result.found == 2  # DataFrame and Block from core/frame.py, nothing from tests
    assert [r["qualname"] for r in result.records] == ["DataFrame"]
