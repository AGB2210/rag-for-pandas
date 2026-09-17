"""Fetch pandas-tagged Stack Overflow questions and their accepted answers.

Raw API responses are saved untouched. Labelling is a separate step so that
labelling rules can change without spending the daily API quota again.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from rag_for_pandas.paths import STACKOVERFLOW_RAW

API_URL = "https://api.stackexchange.com/2.3"
# Anonymous access stops at page 25 ("page above 25 requires access token").
# 25 pages cost about 50 requests: one for questions, one for their answers.
PAGES = 25
PAGE_SIZE = 100
# The anonymous quota is 300 requests per day. Pausing between calls also
# keeps well clear of the per-second throttle.
PAUSE_SECONDS = 1.0


def fetch_cached(client: httpx.Client, path: str, params: dict, cache_file: Path) -> dict:
    """Return the API response for a request, downloading only if not already on disk."""
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    response = client.get(f"{API_URL}{path}", params={"site": "stackoverflow", **params})
    payload = response.json()
    if "error_id" in payload:
        raise RuntimeError(f"API error {payload['error_id']}: {payload.get('error_message')}")
    response.raise_for_status()

    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(PAUSE_SECONDS)
    return payload


def fetch_pages(raw_dir: Path = STACKOVERFLOW_RAW, pages: int = PAGES) -> tuple[int, int, int]:
    """Fetch questions page by page with their accepted answers.

    Returns (questions fetched, questions with an accepted answer, answers fetched).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    questions_fetched = accepted_total = answers_fetched = 0

    with httpx.Client(timeout=30) as client:
        for page in range(1, pages + 1):
            payload = fetch_cached(
                client,
                "/questions",
                {"tagged": "pandas", "sort": "votes", "order": "desc",
                 "pagesize": PAGE_SIZE, "page": page, "filter": "withbody"},
                raw_dir / f"questions_page{page}.json",
            )
            questions_fetched += len(payload["items"])

            # One answers file per questions page, so each cached file always
            # matches the same questions however many pages are fetched later.
            answer_ids = [q["accepted_answer_id"] for q in payload["items"] if "accepted_answer_id" in q]
            accepted_total += len(answer_ids)
            if not answer_ids:
                continue
            answers = fetch_cached(
                client,
                f"/answers/{';'.join(str(i) for i in answer_ids)}",
                {"filter": "withbody", "pagesize": PAGE_SIZE},
                raw_dir / f"answers_page{page}.json",
            )
            answers_fetched += len(answers["items"])

    return questions_fetched, accepted_total, answers_fetched


def load_items(pattern: str, raw_dir: Path = STACKOVERFLOW_RAW) -> list[dict]:
    """All items from the cached API responses matching a glob such as 'questions_page*.json'."""
    items: list[dict] = []
    for path in sorted(raw_dir.glob(pattern)):
        items.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    return items
