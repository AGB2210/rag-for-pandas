import pytest

from rag_for_pandas.gold import gold_queries, next_round, sample_round


def test_gold_queries_builds_lenient_and_strict_labels_and_skips_unanswerable():
    rows = [
        {"query": "How to use dropna", "primary": "dropna", "also_correct": "isna notna", "answerable": "yes"},
        {"query": "Plot with seaborn", "primary": "", "also_correct": "", "answerable": "no"},
    ]
    lenient, strict = gold_queries(rows)

    assert lenient == [{"query": "How to use dropna", "label_in_query": True, "labels": ["dropna", "isna", "notna"]}]
    assert strict == [{"query": "How to use dropna", "label_in_query": True, "labels": ["dropna"]}]


def test_next_round_follows_the_round_sizes():
    rounds = [(100, 7), (200, 8)]
    assert next_round(0, rounds) == (100, 7)
    assert next_round(100, rounds) == (200, 8)


@pytest.mark.parametrize("rows_done", [50, 300])
def test_next_round_refuses_counts_that_do_not_start_a_round(rows_done):
    with pytest.raises(ValueError):
        next_round(rows_done, [(100, 7), (200, 8)])


def test_sample_round_excludes_taken_and_unanswered_questions():
    questions = [{"question_id": i, "accepted_answer_id": i * 10} for i in range(1, 11)]
    questions += [{"question_id": 11}, {"question_id": 2, "accepted_answer_id": 20}]  # unanswered, duplicate

    sample = sample_round(questions, taken={1, 2, 3}, size=4, seed=5)

    ids = [q["question_id"] for q in sample]
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) <= set(range(4, 11))
    assert sample == sample_round(questions, taken={1, 2, 3}, size=4, seed=5)
