"""Build the documentation corpus from a pandas source checkout.

Usage: python scripts/build_corpus.py
"""

from __future__ import annotations

from rag_for_pandas import paths
from rag_for_pandas.corpus import extract
from rag_for_pandas.jsonl import write_jsonl


def main() -> None:
    result = extract(paths.PANDAS_SOURCE)
    records = result.records
    paths.CORPUS.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.CORPUS, records)

    total_words = sum(r["word_count"] for r in records)
    print(f"files parsed:     {result.files_parsed}")
    print(f"files failed:     {result.files_failed}")
    print(f"docstrings found: {result.found}")
    print(f"dropped internal: {result.found - len(records)}")
    print(f"docstrings kept:  {len(records)}")
    print(f"total words:      {total_words:,}")
    print(f"median words:     {sorted(r['word_count'] for r in records)[len(records) // 2]}")
    print(f"written to:       {paths.CORPUS}")


if __name__ == "__main__":
    main()
