import json

import httpx
import pytest

from rag_for_pandas import stackoverflow
from rag_for_pandas.stackoverflow import fetch_cached, load_items


def client_answering(payload, status=200, calls=None):
    """An httpx client whose requests never leave the process."""

    def handle(request):
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(stackoverflow.time, "sleep", lambda seconds: None)


def test_fetch_cached_downloads_once_then_reads_the_file(tmp_path):
    cache = tmp_path / "questions_page1.json"
    calls = []
    with client_answering({"items": [{"question_id": 1}]}, calls=calls) as client:
        first = fetch_cached(client, "/questions", {"page": 1}, cache)
        second = fetch_cached(client, "/questions", {"page": 1}, cache)

    assert first == second == {"items": [{"question_id": 1}]}
    assert len(calls) == 1
    assert "site=stackoverflow" in calls[0] and "page=1" in calls[0]
    assert json.loads(cache.read_text(encoding="utf-8")) == first


def test_fetch_cached_raises_on_an_api_error_and_caches_nothing(tmp_path):
    cache = tmp_path / "questions_page26.json"
    error = {"error_id": 400, "error_message": "page above 25 requires access token"}
    with client_answering(error, status=400) as client, pytest.raises(RuntimeError, match="page above 25"):
        fetch_cached(client, "/questions", {"page": 26}, cache)
    assert not cache.exists()


def test_load_items_joins_matching_files_in_name_order(tmp_path):
    for name, ids in [("questions_page2.json", [3]), ("questions_page1.json", [1, 2]), ("answers_page1.json", [9])]:
        (tmp_path / name).write_text(json.dumps({"items": [{"id": i} for i in ids]}), encoding="utf-8")

    assert [item["id"] for item in load_items("questions_page*.json", tmp_path)] == [1, 2, 3]
