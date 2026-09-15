from docsearch.splits import split_silver

SILVER = [{"question_id": i, "query": f"question {i}"} for i in range(1, 21)] + [
    {"question_id": 99, "query": "How to Drop NA?"}
]
GOLD_ROWS = [
    {"question_id": "3", "query": "question 3"},  # same id
    {"question_id": "500", "query": "how to drop na"},  # same question under another id
]


def test_gold_questions_are_removed_by_id_and_by_title():
    split = split_silver(SILVER, GOLD_ROWS, fraction=0.25, seed=1)
    kept_ids = {q["question_id"] for q in split.train + split.validation}

    assert split.removed == 2
    assert kept_ids.isdisjoint({3, 99, 500})
    assert len(kept_ids) == 19


def test_split_sizes_have_no_overlap_and_are_reproducible():
    first = split_silver(SILVER, GOLD_ROWS, fraction=0.25, seed=1)
    second = split_silver(list(reversed(SILVER)), GOLD_ROWS, fraction=0.25, seed=1)

    assert len(first.validation) == round(19 * 0.25)
    assert {q["question_id"] for q in first.train}.isdisjoint(q["question_id"] for q in first.validation)
    assert first == second  # input order does not change the split
