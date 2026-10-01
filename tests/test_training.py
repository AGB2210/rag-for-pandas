import numpy as np

from rag_for_pandas.corpus import document_text
from rag_for_pandas.training import (
    MAX_LABELS,
    NEGATIVE_SKIP_TOP,
    Prefixes,
    add_hard_negatives,
    preferred_documents,
    training_pairs,
    validation_hits,
)


def doc(qualname: str) -> dict:
    return {"qualname": qualname, "docstring": f"docs for {qualname}"}


def test_preferred_documents_picks_the_most_used_owner():
    docs = [doc("Series.sum"), doc("GroupBy.sum"), doc("DataFrame.sum"), doc("read_csv"), doc("Styler.format")]
    preferred = preferred_documents(docs)

    assert preferred["sum"]["qualname"] == "DataFrame.sum"
    assert preferred["read_csv"]["qualname"] == "read_csv"
    assert preferred["format"]["qualname"] == "Styler.format"  # only owner, even if not in the preference list


def test_training_pairs_one_per_label_and_skip_questions_with_many_labels():
    preferred = preferred_documents([doc(f"DataFrame.m{i}") for i in range(MAX_LABELS + 1)])
    queries = [
        {"query": "two labels", "labels": ["m0", "m1"]},
        {"query": "too many labels", "labels": [f"m{i}" for i in range(MAX_LABELS + 1)]},
    ]
    pairs, skipped = training_pairs(preferred, queries)

    assert skipped == 1
    assert [(p["anchor"], p["positive"]) for p in pairs] == [
        ("two labels", document_text(preferred["m0"])),
        ("two labels", document_text(preferred["m1"])),
    ]


class FakeEncoder:
    """Returns fixed vectors so the ranking is known in advance."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, sentences, normalize_embeddings=True, batch_size=64):
        return np.array([self.vectors[s] for s in sentences], dtype=float)


def test_hard_negatives_skip_the_top_ranks_and_never_use_a_labelled_answer():
    names = ["a", "b", "c", "d", "e"]
    preferred = preferred_documents([doc(n) for n in names])
    # One-hot documents and a query scoring them a > b > c > d > e.
    vectors = {document_text(preferred[n]): np.eye(5)[i].tolist() for i, n in enumerate(names)}
    vectors["question"] = [5, 4, 3, 2, 1]
    pairs = [
        {"anchor": "question", "positive": document_text(preferred["a"]), "labels": ["a"]},
        {"anchor": "question", "positive": document_text(preferred["d"]), "labels": ["d"]},
    ]

    add_hard_negatives(FakeEncoder(vectors), pairs, preferred)

    assert NEGATIVE_SKIP_TOP == 3  # the expectations below assume a, b and c are skipped
    assert pairs[0]["negative"] == document_text(preferred["d"])
    assert pairs[1]["negative"] == document_text(preferred["e"])  # d is a labelled answer


def test_hard_negatives_can_skip_fewer_ranks_and_carry_a_document_prefix():
    names = ["a", "b", "c", "d", "e"]
    preferred = preferred_documents([doc(n) for n in names])
    vectors = {"passage: " + document_text(preferred[n]): np.eye(5)[i].tolist() for i, n in enumerate(names)}
    vectors["query: question"] = [5, 4, 3, 2, 1]
    pairs = [{"anchor": "query: question", "positive": "passage: " + document_text(preferred["a"]), "labels": ["a"]}]

    add_hard_negatives(FakeEncoder(vectors), pairs, preferred, skip_top=1, document_prefix="passage: ")

    assert pairs[0]["negative"] == "passage: " + document_text(preferred["b"])  # rank 2: only rank 1 is skipped


def test_training_pairs_add_the_prefixes_a_model_expects():
    preferred = preferred_documents([doc("DataFrame.dropna")])
    pairs, _ = training_pairs(preferred, [{"query": "drop empty rows", "labels": ["dropna"]}], Prefixes("query: ", "passage: "))

    assert pairs[0]["anchor"] == "query: drop empty rows"
    assert pairs[0]["positive"] == "passage: " + document_text(preferred["dropna"])


def test_validation_hits_mark_each_question_at_each_depth():
    docs = [doc(f"DataFrame.m{i}") for i in range(12)]
    vectors = {document_text(d): np.eye(12)[i].tolist() for i, d in enumerate(docs)}
    vectors["first"] = np.arange(12, 0, -1).tolist()  # ranks m0, m1, ... m11
    vectors["second"] = np.arange(12, 0, -1).tolist()
    queries = [{"query": "first", "labels": ["m0"]}, {"query": "second", "labels": ["m7"]}]

    assert validation_hits(FakeEncoder(vectors), docs, queries) == {"R@1": [1, 0], "R@5": [1, 0], "R@10": [1, 1]}
