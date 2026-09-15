import numpy as np

from docsearch.corpus import document_text
from docsearch.reranker_training import reranker_pairs
from docsearch.training import MAX_LABELS, preferred_documents

DOCS = [
    {"qualname": q, "docstring": f"docs for {q}"}
    for q in ["merge", "merge_ordered", "DataFrame.join", "Series.sum", "DataFrame.sum", "concat", "DataFrame.align"]
]
PREFERRED = preferred_documents(DOCS)


def labels_by_response(pairs):
    return {(p["response"].split(":")[0], p["label"]) for p in pairs}


def test_positives_come_from_labels_and_negatives_from_retrieved_wrong_answers():
    queries = [{"query": "merge on a column", "labels": ["merge"]}]
    rankings = [np.array([0, 1, 2, 5, 6])]  # merge, merge_ordered, join, concat, align

    pairs = reranker_pairs(DOCS, PREFERRED, queries, rankings, skip_top=0, negatives_per_question=2)

    assert pairs[0] == {"query": "merge on a column", "response": document_text(PREFERRED["merge"]), "label": 1}
    assert labels_by_response(pairs) == {("merge", 1), ("merge_ordered", 0), ("DataFrame.join", 0)}


def test_skip_top_and_same_name_documents_are_never_negatives():
    queries = [{"query": "sum columns", "labels": ["sum"]}]
    rankings = [np.array([3, 1, 4, 2, 5])]  # Series.sum, merge_ordered, DataFrame.sum, join, concat

    pairs = reranker_pairs(DOCS, PREFERRED, queries, rankings, skip_top=2, negatives_per_question=4)

    negatives = {name for name, label in labels_by_response(pairs) if label == 0}
    assert negatives == {"DataFrame.join", "concat"}  # merge_ordered skipped; no document named sum


def test_questions_with_too_many_labels_are_skipped():
    labels = ["merge", "merge_ordered", "concat", "sum"][: MAX_LABELS + 1]
    pairs = reranker_pairs(DOCS, PREFERRED, [{"query": "q", "labels": labels}], [np.array([0, 1])], skip_top=0)
    assert pairs == []
