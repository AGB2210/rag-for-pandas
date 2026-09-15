import numpy as np

from docsearch.corpus import document_text
from docsearch.retrieval import EmbeddingRetriever, bm25_rankings, top_k
from docsearch.text import normalise_title, tokenize


def test_top_k_orders_documents_by_cosine_score():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])

    order, scores = top_k(queries, docs, depth=2)

    assert order.tolist() == [[0, 1], [2, 1]]
    assert np.allclose(scores, [[1.0, 0.6], [1.0, 0.8]])


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def encode(self, sentences, normalize_embeddings=True, batch_size=64):
        self.calls += 1
        return np.array([self.vectors[s] for s in sentences], dtype=float)


def test_embedding_retriever_encodes_documents_once_and_searches():
    docs = [{"qualname": "dropna", "docstring": "remove"}, {"qualname": "fillna", "docstring": "fill"}]
    encoder = FakeEncoder({document_text(docs[0]): [1.0, 0.0], document_text(docs[1]): [0.0, 1.0], "fill gaps": [0.28, 0.96]})

    retriever = EmbeddingRetriever(docs, encoder)
    first = retriever.search("fill gaps", k=2)
    retriever.search("fill gaps", k=1)

    assert [index for index, _ in first] == [1, 0]
    assert round(first[0][1], 2) == 0.96
    assert encoder.calls == 3  # documents once, then one call per search


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
