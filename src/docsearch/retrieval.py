"""Search methods. Each returns, per query, corpus indices ordered best first."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from docsearch.corpus import document_text
from docsearch.paths import EMBEDDINGS_CACHE
from docsearch.text import tokenize

BASE_MODEL = "all-MiniLM-L6-v2"


def bm25_rankings(docs: list[dict], queries: list[dict], depth: int) -> list[np.ndarray]:
    index = BM25Okapi([tokenize(document_text(d)) for d in docs])
    return [np.argsort(-index.get_scores(tokenize(q["query"])))[:depth] for q in queries]


def embedding_rankings(
    docs: list[dict],
    queries: list[dict],
    depth: int,
    model_name: str = BASE_MODEL,
    cache: Path | None = EMBEDDINGS_CACHE,
) -> list[np.ndarray]:
    """Rank by cosine similarity of normalised embeddings.

    Pass cache=None for a model that gets retrained: the size check below
    cannot tell old vectors from new ones when the corpus is unchanged.
    """
    model = SentenceTransformer(model_name)
    doc_vectors = np.load(cache) if cache is not None and cache.exists() else None
    # A cache built from a different corpus would silently return wrong results.
    if doc_vectors is None or doc_vectors.shape[0] != len(docs):
        doc_vectors = model.encode([document_text(d) for d in docs], normalize_embeddings=True, batch_size=64)
        if cache is not None:
            np.save(cache, doc_vectors)

    query_vectors = model.encode([q["query"] for q in queries], normalize_embeddings=True)
    scores = query_vectors @ doc_vectors.T
    return [np.argsort(-row)[:depth] for row in scores]
