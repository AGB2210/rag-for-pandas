import pytest
from fastapi.testclient import TestClient

from docsearch.api import MAX_QUERY_CHARS, MAX_RESULTS, create_app
from docsearch.generation import NOT_FOUND
from docsearch.pipeline import Pipeline

DOCS = [
    {"qualname": "DataFrame.fillna", "docstring": "Fill NA/NaN values.", "source_file": "core/generic.py"},
    {"qualname": "DataFrame.dropna", "docstring": "Remove missing values.", "source_file": "core/frame.py"},
    {"qualname": "read_csv", "docstring": "Read a comma-separated values file.", "source_file": "io/parsers.py"},
]


class FakeRetriever:
    """Always ranks dropna, then fillna, then read_csv."""

    def search(self, query, k):
        return [(1, 0.91), (0, 0.52), (2, 0.1)][:k]


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return self.answer


def make_client(generator):
    app = create_app(lambda: Pipeline(DOCS, FakeRetriever(), generator))
    return TestClient(app)


@pytest.fixture
def client():
    # Entering the client runs the app's startup, which loads the pipeline.
    with make_client(FakeGenerator("Use DataFrame.dropna to remove them [1]. See also [7].")) as test_client:
        yield test_client


def test_health_reports_documents_and_answer_support(client):
    assert client.get("/health").json() == {"status": "ok", "documents": 3, "answers_enabled": True}


def test_search_returns_ranked_results(client):
    response = client.post("/search", json={"query": "delete empty rows", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "delete empty rows"
    assert [(r["rank"], r["name"], r["score"]) for r in body["results"]] == [(1, "DataFrame.dropna", 0.91), (2, "DataFrame.fillna", 0.52)]
    assert body["results"][0]["excerpt"] == "DataFrame.dropna: Remove missing values."
    assert body["results"][0]["source_file"] == "core/frame.py"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * (MAX_QUERY_CHARS + 1)},
        {"query": "ok", "k": 0},
        {"query": "ok", "k": MAX_RESULTS + 1},
        {},
    ],
)
def test_search_rejects_invalid_requests(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_answer_marks_cited_sources_and_invalid_citations(client):
    response = client.post("/answer", json={"question": "  delete empty rows  "})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "delete empty rows"  # surrounding whitespace removed
    assert body["abstained"] is False
    assert body["sources"] == [
        {"number": 1, "name": "DataFrame.dropna", "cited": True},
        {"number": 2, "name": "DataFrame.fillna", "cited": False},
        {"number": 3, "name": "read_csv", "cited": False},
    ]
    assert body["invalid_citations"] == [7]


def test_answer_reports_abstention():
    with make_client(FakeGenerator(NOT_FOUND)) as test_client:
        body = test_client.post("/answer", json={"question": "plot with seaborn"}).json()
    assert body["abstained"] is True
    assert not any(source["cited"] for source in body["sources"])


def test_answer_is_unavailable_without_a_generator():
    with make_client(None) as test_client:
        assert test_client.get("/health").json()["answers_enabled"] is False
        response = test_client.post("/answer", json={"question": "delete empty rows"})
        assert response.status_code == 503
        # Search still works without a generator.
        assert test_client.post("/search", json={"query": "delete empty rows"}).status_code == 200


def test_invalid_answer_request_never_reaches_the_generator():
    generator = FakeGenerator("unused")
    with make_client(generator) as test_client:
        assert test_client.post("/answer", json={"question": ""}).status_code == 422
    assert generator.calls == 0
