"""Re-order a retriever's top results with a cross-encoder.

The retriever embeds question and document separately, which is fast enough
to search the whole corpus but never lets the model read both texts together.
A cross-encoder reads each (question, document) pair jointly and scores it,
which is more accurate but far slower, so it only re-orders the top results.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_for_pandas.corpus import document_text

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class PairScorer(Protocol):
    def predict(self, inputs: list[tuple[str, str]], batch_size: int) -> np.ndarray: ...


def rerank(
    scorer: PairScorer, docs: list[dict], queries: list[dict], rankings: list[np.ndarray], depth: int
) -> list[np.ndarray]:
    """Re-order the first `depth` results of each ranking by pair score; later results keep their order."""
    pairs = [(query["query"], document_text(docs[i])) for query, ranking in zip(queries, rankings) for i in ranking[:depth]]
    # One call for all pairs lets the scorer batch across queries.
    scores = scorer.predict(pairs, batch_size=64)

    reranked, start = [], 0
    for ranking in rankings:
        head = ranking[:depth]
        head_scores = scores[start : start + len(head)]
        start += len(head)
        # Stable sort keeps the retriever's order for tied scores.
        reranked.append(np.concatenate([head[np.argsort(-head_scores, kind="stable")], ranking[depth:]]))
    return reranked
