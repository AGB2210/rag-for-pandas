import numpy as np
import pytest

from rag_for_pandas.metrics import hits_at_k, paired_bootstrap_ci, recall_at_k

DOCS = [{"qualname": "DataFrame.dropna"}, {"qualname": "Series.fillna"}, {"qualname": "read_csv"}]
QUERIES = [{"labels": ["dropna"]}, {"labels": ["dropna"]}]
RANKINGS = [np.array([1, 0, 2]), np.array([2, 1, 0])]


@pytest.mark.parametrize(("k", "expected"), [(1, [0, 0]), (2, [1, 0]), (3, [1, 1])])
def test_hits_at_k_matches_on_final_name(k, expected):
    assert hits_at_k(RANKINGS, DOCS, QUERIES, k).tolist() == expected


def test_recall_at_k_is_the_mean_hit_rate():
    assert recall_at_k(RANKINGS, DOCS, QUERIES, 2) == 0.5


def test_bootstrap_of_identical_results_is_zero():
    results = np.array([1.0, 0.0] * 50)
    assert paired_bootstrap_ci(results, results) == (0.0, 0.0)


def test_bootstrap_of_a_constant_gap_is_that_gap():
    assert paired_bootstrap_ci(np.zeros(100), np.ones(100)) == (1.0, 1.0)


def test_bootstrap_interval_contains_the_observed_gap_and_is_reproducible():
    rng = np.random.default_rng(1)
    baseline = rng.integers(0, 2, 200).astype(float)
    candidate = rng.integers(0, 2, 200).astype(float)
    low, high = paired_bootstrap_ci(baseline, candidate)
    assert low <= (candidate - baseline).mean() <= high
    assert paired_bootstrap_ci(baseline, candidate) == (low, high)
