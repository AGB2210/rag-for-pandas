import numpy as np

from rag_for_pandas.reranking import rerank

DOCS = [{"qualname": f"d{i}", "docstring": f"text {i}"} for i in range(5)]


class FakeScorer:
    """Scores a pair by a fixed number per document, and records what it was asked."""

    def __init__(self, score_by_doc: dict[str, float]):
        self.score_by_doc = score_by_doc
        self.seen: list[tuple[str, str]] = []

    def predict(self, inputs, batch_size=64):
        self.seen.extend(inputs)
        return np.array([self.score_by_doc[text.split(":")[0]] for _, text in inputs])


def test_rerank_reorders_only_the_top_results():
    scorer = FakeScorer({"d0": 0.1, "d1": 0.9, "d2": 0.5, "d3": 5.0, "d4": 9.0})
    rankings = [np.array([0, 1, 2, 3, 4])]

    reranked = rerank(scorer, DOCS, [{"query": "q"}], rankings, depth=3)

    assert reranked[0].tolist() == [1, 2, 0, 3, 4]  # d3 and d4 stay below despite high scores
    assert len(scorer.seen) == 3


def test_rerank_handles_several_queries_in_one_batch_and_keeps_ties_stable():
    scorer = FakeScorer({"d0": 1.0, "d1": 1.0, "d2": 2.0, "d3": 0.0, "d4": 0.0})
    queries = [{"query": "first"}, {"query": "second"}]
    rankings = [np.array([0, 1, 2]), np.array([4, 3, 2])]

    reranked = rerank(scorer, DOCS, queries, rankings, depth=3)

    assert reranked[0].tolist() == [2, 0, 1]  # d0 and d1 tie and keep retriever order
    assert reranked[1].tolist() == [2, 4, 3]
    assert [query for query, _ in scorer.seen] == ["first"] * 3 + ["second"] * 3
