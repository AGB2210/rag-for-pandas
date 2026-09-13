"""Keyword search over the pandas docstrings. This is our baseline to beat."""

import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

DOCS_PATH = Path("data/interim/docstrings.jsonl")


def load_docs() -> list[dict]:
    with DOCS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words. Crude on purpose: this is the baseline."""
    return text.lower().split()


def preview(docstring: str, width: int = 90) -> str:
    return " ".join(docstring.split())[:width]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "merge two dataframes on different column names"

    docs = load_docs()
    index = BM25Okapi([tokenize(d["docstring"]) for d in docs])

    scores = index.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    print(f"QUERY: {query}\n")
    for rank, i in enumerate(top, start=1):
        print(f"{rank}. {docs[i]['qualname']}   (score {scores[i]:.1f})")
        print(f"   {preview(docs[i]['docstring'])}\n")


if __name__ == "__main__":
    main()
