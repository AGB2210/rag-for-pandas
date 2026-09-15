from docsearch.retrieval import bm25_rankings
from docsearch.text import normalise_title, tokenize


def test_tokenize_splits_qualified_names_and_lowercases():
    assert tokenize("DataFrame.dropna(axis=1)") == ["dataframe", "dropna", "axis", "1"]


def test_normalise_title_ignores_case_and_punctuation():
    assert normalise_title("How to Drop NA?") == normalise_title("how to drop na")


def test_bm25_ranks_the_document_sharing_rare_words_first():
    docs = [
        {"qualname": "DataFrame.plot", "docstring": "Make plots of a DataFrame."},
        {"qualname": "DataFrame.dropna", "docstring": "Remove missing values."},
        {"qualname": "read_csv", "docstring": "Read a comma-separated values file into a DataFrame."},
    ]
    rankings = bm25_rankings(docs, [{"query": "remove missing values"}], depth=3)
    assert rankings[0][0] == 1
