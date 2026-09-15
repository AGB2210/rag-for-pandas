import numpy as np

from docsearch.corpus import document_text
from docsearch.training import MAX_LABELS, NEGATIVE_SKIP_TOP, add_hard_negatives, preferred_documents, training_pairs


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
