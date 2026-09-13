"""Measure BM25 and embedding search on the Stack Overflow evaluation set."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("data/interim/docstrings.jsonl")
QUERIES_PATH = Path("data/eval/queries.jsonl")
GOLD_PATH = Path("annotations/gold_queries.csv")
EMBEDDINGS_CACHE = Path("data/interim/embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"
K_VALUES = (1, 5, 10)
CI_K = 5
BOOTSTRAP_SAMPLES = 10_000
SEED = 0


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


def hits_at_k(rankings: list[np.ndarray], docs: list[dict], queries: list[dict], k: int) -> np.ndarray:
    """1.0 for each query with at least one correct name in the top k results, else 0.0."""
    return np.array(
        [
            bool({docs[i]["qualname"].split(".")[-1] for i in ranking[:k]} & set(query["labels"]))
            for ranking, query in zip(rankings, queries)
        ],
        dtype=float,
    )


def recall_at_k(rankings: list[np.ndarray], docs: list[dict], queries: list[dict], k: int) -> float:
    """Fraction of queries where at least one correct name is in the top k results."""
    return float(hits_at_k(rankings, docs, queries, k).mean())


def paired_bootstrap_ci(baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """95% interval for the mean per-query gap `candidate - baseline`.

    Resamples queries with replacement. It is paired: each resample keeps both
    methods' results for the same queries, so how hard a query is cancels out
    and only the difference between the methods varies.
    """
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(baseline), size=(BOOTSTRAP_SAMPLES, len(baseline)))
    gaps = (candidate[picks] - baseline[picks]).mean(axis=1)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(low), float(high)


def load_gold() -> tuple[list[dict], list[dict]]:
    """Hand-labelled queries, as lenient (any correct name) and strict (primary only) copies.

    Questions marked unanswerable are left out: no document can be retrieved
    for them, so they would only lower every method's score by the same amount.
    """
    with GOLD_PATH.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["answerable"] == "yes"]
    lenient, strict = [], []
    for row in rows:
        query_words = set(re.findall(r"\w+", row["query"].lower()))
        base = {"query": row["query"], "label_in_query": row["primary"].lower() in query_words}
        lenient.append({**base, "labels": [row["primary"], *row["also_correct"].split()]})
        strict.append({**base, "labels": [row["primary"]]})
    return lenient, strict


def report(title: str, docs: list[dict], queries: list[dict], methods: dict[str, list[np.ndarray]]) -> None:
    subsets = {
        "ALL QUERIES": list(range(len(queries))),
        "NAME IN TITLE (easy)": [i for i, q in enumerate(queries) if q["label_in_query"]],
        "NAME NOT IN TITLE (honest test)": [i for i, q in enumerate(queries) if not q["label_in_query"]],
    }
    print(f"\n===== {title} =====")
    for subset_name, indices in subsets.items():
        subset_queries = [queries[i] for i in indices]
        print(f"\n{subset_name}  (n={len(indices)})")
        print(f"  {'method':<12}" + "".join(f"{'R@' + str(k):>8}" for k in K_VALUES))
        for method_name, rankings in methods.items():
            subset_rankings = [rankings[i] for i in indices]
            values = [recall_at_k(subset_rankings, docs, subset_queries, k) for k in K_VALUES]
            print(f"  {method_name:<12}" + "".join(f"{v:>8.2f}" for v in values))
        baseline, candidate = (
            hits_at_k([methods[name][i] for i in indices], docs, subset_queries, CI_K)
            for name in ("BM25", "Embeddings")
        )
        low, high = paired_bootstrap_ci(baseline, candidate)
        gap = candidate.mean() - baseline.mean()
        print(f"  Embeddings - BM25 at R@{CI_K}: {gap:+.2f}   95% CI [{low:+.2f}, {high:+.2f}]")


def main() -> None:
    docs = load_jsonl(CORPUS_PATH)
    depth = max(K_VALUES)
    silver = load_jsonl(QUERIES_PATH)
    gold_lenient, gold_strict = load_gold()

    for title, queries in [
        ("SILVER: automatic labels", silver),
        ("GOLD: any correct name", gold_lenient),
        ("GOLD: primary name only", gold_strict),
    ]:
        methods = {
            "BM25": bm25_rankings(docs, queries, depth),
            "Embeddings": embedding_rankings(docs, queries, depth),
        }
        report(title, docs, queries, methods)


if __name__ == "__main__":
    main()
