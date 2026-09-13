"""Measure BM25 and embedding search on the Stack Overflow evaluation set."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("data/interim/docstrings.jsonl")
QUERIES_PATH = Path("data/eval/queries.jsonl")
EMBEDDINGS_CACHE = Path("data/interim/embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"
K_VALUES = (1, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def document_text(doc: dict) -> str:
    # Both methods must see identical text; otherwise the comparison measures
    # the input rather than the search method.
    return f"{doc['qualname']}: {doc['docstring']}"


def tokenize(text: str) -> list[str]:
    # \w+ splits "DataFrame.dropna" into "dataframe" and "dropna" so a query
    # containing "dropna" can match. A sloppy tokenizer would weaken the
    # baseline and exaggerate every later improvement.
    return re.findall(r"\w+", text.lower())


def bm25_rankings(docs: list[dict], queries: list[dict], depth: int) -> list[np.ndarray]:
    index = BM25Okapi([tokenize(document_text(d)) for d in docs])
    return [np.argsort(-index.get_scores(tokenize(q["query"])))[:depth] for q in queries]


def embedding_rankings(docs: list[dict], queries: list[dict], depth: int) -> list[np.ndarray]:
    model = SentenceTransformer(MODEL_NAME)
    doc_vectors = np.load(EMBEDDINGS_CACHE) if EMBEDDINGS_CACHE.exists() else None
    # A cache built from a different corpus would silently return wrong results.
    if doc_vectors is None or doc_vectors.shape[0] != len(docs):
        doc_vectors = model.encode(
            [document_text(d) for d in docs], normalize_embeddings=True, batch_size=64
        )
        np.save(EMBEDDINGS_CACHE, doc_vectors)

    query_vectors = model.encode([q["query"] for q in queries], normalize_embeddings=True)
    scores = query_vectors @ doc_vectors.T
    return [np.argsort(-row)[:depth] for row in scores]


def recall_at_k(rankings: list[np.ndarray], docs: list[dict], queries: list[dict], k: int) -> float:
    """Fraction of queries where at least one correct name is in the top k results."""
    hits = 0
    for ranking, query in zip(rankings, queries):
        retrieved_names = {docs[i]["qualname"].split(".")[-1] for i in ranking[:k]}
        hits += bool(retrieved_names & set(query["labels"]))
    return hits / len(queries)


def main() -> None:
    docs = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    depth = max(K_VALUES)

    methods = {
        "BM25": bm25_rankings(docs, queries, depth),
        "Embeddings": embedding_rankings(docs, queries, depth),
    }
    subsets = {
        "ALL QUERIES": list(range(len(queries))),
        "NAME IN TITLE (easy)": [i for i, q in enumerate(queries) if q["label_in_query"]],
        "NAME NOT IN TITLE (honest test)": [i for i, q in enumerate(queries) if not q["label_in_query"]],
    }

    for subset_name, indices in subsets.items():
        subset_queries = [queries[i] for i in indices]
        print(f"\n{subset_name}  (n={len(indices)})")
        print(f"  {'method':<12}" + "".join(f"{'R@' + str(k):>8}" for k in K_VALUES))
        for method_name, rankings in methods.items():
            subset_rankings = [rankings[i] for i in indices]
            values = [recall_at_k(subset_rankings, docs, subset_queries, k) for k in K_VALUES]
            print(f"  {method_name:<12}" + "".join(f"{v:>8.2f}" for v in values))


if __name__ == "__main__":
    main()
