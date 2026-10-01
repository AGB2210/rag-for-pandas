import pytest
from compare_runs import mean_hits, summary_lines


def run(hits_at_5: list[int], question_ids: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    return {"question_ids": list(question_ids), "R@1": [0] * len(hits_at_5), "R@5": hits_at_5, "R@10": [1] * len(hits_at_5)}


def test_mean_hits_averages_runs_per_question():
    assert mean_hits([run([1, 0, 1, 0]), run([1, 1, 0, 0])], "R@5").tolist() == [1.0, 0.5, 0.5, 0.0]


def test_summary_reports_each_group_and_the_gap_to_the_first():
    groups = [
        ("skip 3", [run([1, 0, 0, 0]), run([1, 1, 0, 0])]),
        ("skip 1", [run([1, 1, 1, 0]), run([1, 1, 1, 0])]),
    ]
    lines = summary_lines(groups)

    assert lines[0] == "validation questions: 4"
    assert lines[2].split() == ["skip", "3", "2", "0.000", "0.375", "1.000", "0.250,", "0.500"]
    assert lines[3].split()[:6] == ["skip", "1", "2", "0.000", "0.750", "1.000"]
    assert lines[4].startswith("  skip 1 - skip 3 at R@5: +0.375   95% CI [")


def test_summary_refuses_runs_scored_on_different_questions():
    groups = [("a", [run([1, 0, 0, 0])]), ("b", [run([1, 0, 0, 0], question_ids=(1, 2, 3, 9))])]
    with pytest.raises(ValueError, match="different questions"):
        summary_lines(groups)
