"""Retrieval metrics and the confidence interval used to compare methods."""

from __future__ import annotations

import numpy as np

from rag_for_pandas.corpus import final_name

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def hits_at_k(rankings: list[np.ndarray], docs: list[dict], queries: list[dict], k: int) -> np.ndarray:
    """1.0 for each query with at least one correct name in the top k results, else 0.0."""
    return np.array(
        [
            bool({final_name(docs[i]["qualname"]) for i in ranking[:k]} & set(query["labels"]))
            for ranking, query in zip(rankings, queries)
        ],
        dtype=float,
    )


def recall_at_k(rankings: list[np.ndarray], docs: list[dict], queries: list[dict], k: int) -> float:
    """Fraction of queries where at least one correct name is in the top k results."""
    return float(hits_at_k(rankings, docs, queries, k).mean())


def paired_bootstrap_ci(
    baseline: np.ndarray, candidate: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """95% interval for the mean per-query gap `candidate - baseline`.

    Resamples queries with replacement. It is paired: each resample keeps both
    methods' results for the same queries, so how hard a query is cancels out
    and only the difference between the methods varies.
    """
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    gaps = (candidate[picks] - baseline[picks]).mean(axis=1)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(low), float(high)
