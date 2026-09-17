import numpy as np

from rag_for_pandas.evaluation import report_lines, subsets

DOCS = [{"qualname": "DataFrame.dropna"}, {"qualname": "DataFrame.fillna"}, {"qualname": "read_csv"}]
QUERIES = [
    {"labels": ["dropna"], "label_in_query": True},
    {"labels": ["fillna"], "label_in_query": False},
    {"labels": ["read_csv"], "label_in_query": False},
]


def test_subsets_split_queries_by_whether_the_name_is_in_the_title():
    assert subsets(QUERIES) == {
        "ALL QUERIES": [0, 1, 2],
        "NAME IN TITLE (easy)": [0],
        "NAME NOT IN TITLE (honest test)": [1, 2],
    }


def test_report_lines_show_recall_per_subset_and_compare_methods_in_order():
    methods = {
        # Correct document first only for the easy query.
        "First": [np.array([0, 1, 2]), np.array([0, 2, 1]), np.array([0, 1, 2])],
        # Correct document first for every query.
        "Second": [np.array([0, 1, 2]), np.array([1, 0, 2]), np.array([2, 0, 1])],
        "Third": [np.array([0, 1, 2]), np.array([1, 0, 2]), np.array([2, 0, 1])],
    }
    lines = report_lines("TITLE", DOCS, QUERIES, methods)

    assert "===== TITLE =====" in lines
    honest = lines[lines.index("NAME NOT IN TITLE (honest test)  (n=2)") :]
    # R@1, R@5 and R@10 on the honest subset.
    assert honest[2].split() == ["First", "0.00", "1.00", "1.00"]
    assert honest[3].split() == ["Second", "1.00", "1.00", "1.00"]
    # Each method against the previous one, then the last against the first, all at R@5.
    comparisons = [line.split(" at R@5")[0].strip() for line in honest[5:8]]
    assert comparisons == ["Second - First", "Third - Second", "Third - First"]


def test_report_lines_skip_the_last_against_first_comparison_for_two_methods():
    rankings = [np.array([0, 1, 2])] * 3
    lines = report_lines("TITLE", DOCS, QUERIES, {"A": rankings, "B": rankings})
    assert sum("B - A at R@5" in line for line in lines) == 3  # one per subset
    assert not any("A - B" in line for line in lines)
