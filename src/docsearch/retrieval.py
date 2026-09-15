"""Search methods. Each returns, per query, corpus indices ordered best first."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from docsearch.corpus import document_text
from docsearch.paths import EMBEDDINGS_CACHE
from docsearch.text import tokenize

BASE_MODEL = "all-MiniLM-L6-v2"


class Encoder(Protocol):
    def encode(self, sentences: list[str], normalize_embeddings: bool, batch_size: int) -> np.ndarray: ...


def bm25_rankings(docs: list[dict], queries: list[dict], depth: int) -> list[np.ndarray]:
    index = BM25Okapi([tokenize(document_text(d)) for d in docs])
    return [np.argsort(-index.get_scores(tokenize(q["query"])))[:depth] for q in queries]


def top_k(query_vectors: np.ndarray, doc_vectors: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the `depth` most similar documents for each query, best first.

    Vectors must be normalised, so the dot product equals cosine similarity.
    """
    scores = query_vectors @ doc_vectors.T
    order = np.argsort(-scores, axis=1)[:, :depth]
    return order, np.take_along_axis(scores, order, axis=1)


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
    order, _ = top_k(query_vectors, doc_vectors, depth)
    return list(order)


class EmbeddingRetriever:
    """Holds a model and the corpus vectors in memory, to answer many queries one at a time.

    Document vectors are computed once when the retriever is created, so each
    search only encodes the query.
    """

    def __init__(self, docs: list[dict], encoder: Encoder) -> None:
        self.encoder = encoder
        self.doc_vectors = encoder.encode([document_text(d) for d in docs], normalize_embeddings=True, batch_size=64)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """(corpus index, cosine score) of the k best documents, best first."""
        query_vector = self.encoder.encode([query], normalize_embeddings=True, batch_size=64)
        order, scores = top_k(query_vector, self.doc_vectors, k)
        return list(zip(order[0].tolist(), scores[0].tolist()))
