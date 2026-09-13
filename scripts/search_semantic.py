"""Semantic search over the pandas docstrings, for comparison against BM25."""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("data/interim/docstrings.jsonl")
CACHE_PATH = Path("data/interim/embeddings.npy")


def load_docs() -> list[dict]:
    with DOCS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def preview(docstring: str, width: int = 90) -> str:
    return " ".join(docstring.split())[:width]


def main() -> None:
    import numpy as np

    query = " ".join(sys.argv[1:]) or "delete empty rows from my table"
    docs = load_docs()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embedding 1,678 docstrings takes a while; cache so repeat searches are instant.
    if CACHE_PATH.exists():
        vectors = np.load(CACHE_PATH)
    else:
        texts = [f"{d['qualname']}: {d['docstring']}" for d in docs]
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        np.save(CACHE_PATH, vectors)

    query_vector = model.encode(query, normalize_embeddings=True)
    scores = vectors @ query_vector

    print(f"QUERY: {query}\n")
    for rank, i in enumerate(np.argsort(-scores)[:5], start=1):
        print(f"{rank}. {docs[i]['qualname']}   (score {scores[i]:.2f})")
        print(f"   {preview(docs[i]['docstring'])}\n")


if __name__ == "__main__":
    main()
