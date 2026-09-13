"""Fetch pandas-tagged Stack Overflow questions and their accepted answers.

Raw API responses are saved untouched. Labelling is a separate step so that
labelling rules can change without spending the daily API quota again.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

API_URL = "https://api.stackexchange.com/2.3"
RAW_DIR = Path("data/raw/stackoverflow")
PAGES = 5
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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    questions: list[dict] = []

    with httpx.Client(timeout=30) as client:
        for page in range(1, PAGES + 1):
            payload = fetch_cached(
                client,
                "/questions",
                {"tagged": "pandas", "sort": "votes", "order": "desc",
                 "pagesize": PAGE_SIZE, "page": page, "filter": "withbody"},
                RAW_DIR / f"questions_page{page}.json",
            )
            questions.extend(payload["items"])

        answer_ids = [q["accepted_answer_id"] for q in questions if "accepted_answer_id" in q]

        answers_fetched = 0
        for batch_number, start in enumerate(range(0, len(answer_ids), PAGE_SIZE), start=1):
            ids = ";".join(str(i) for i in answer_ids[start:start + PAGE_SIZE])
            payload = fetch_cached(
                client,
                f"/answers/{ids}",
                {"filter": "withbody", "pagesize": PAGE_SIZE},
                RAW_DIR / f"answers_batch{batch_number}.json",
            )
            answers_fetched += len(payload["items"])

    print(f"questions fetched:          {len(questions)}")
    print(f"with an accepted answer:    {len(answer_ids)}")
    print(f"accepted answers fetched:   {answers_fetched}")


if __name__ == "__main__":
    main()
