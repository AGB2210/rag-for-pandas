"""Draw a random sample of silver labels for manual review.

Writes a CSV with an empty verdict column. The share of wrong labels in a
reviewed sample bounds how far any score measured against them can be trusted.

Usage: python scripts/make_label_review.py
"""

from __future__ import annotations

import csv
import random

from docsearch import paths
from docsearch.jsonl import load_jsonl
from docsearch.labels import evidence_lines
from docsearch.stackoverflow import load_items

SAMPLE_SIZE = 50
SEED = 42
PRINT_FIRST = 10


def main() -> None:
    # A filled-in review is hand-made and cannot be regenerated.
    if paths.LABEL_REVIEW.exists():
        raise SystemExit(f"{paths.LABEL_REVIEW} exists; refusing to overwrite reviewed labels")

    queries = load_jsonl(paths.SILVER_QUERIES)
    accepted = {q["question_id"]: q.get("accepted_answer_id") for q in load_items("questions_page*.json")}
    answers = {a["answer_id"]: a["body"] for a in load_items("answers_page*.json")}

    sample = random.Random(SEED).sample(queries, SAMPLE_SIZE)

    # utf-8-sig adds a byte-order mark so Excel on Windows opens the file with
    # the correct encoding instead of garbling non-ASCII characters.
    with paths.LABEL_REVIEW.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "verdict", "query", "labels", "evidence", "url", "notes"])
        for review_id, record in enumerate(sample, start=1):
            evidence = evidence_lines(answers[accepted[record["question_id"]]], record["labels"])
            writer.writerow(
                [review_id, "", record["query"], " ".join(record["labels"]), " | ".join(evidence), record["url"], ""]
            )
            if review_id <= PRINT_FIRST:
                print(f"[{review_id}] {record['query']}")
                print(f"     label:    {', '.join(record['labels'])}")
                for line in evidence:
                    print(f"     evidence: {line}")
                print()

    print(f"wrote {SAMPLE_SIZE} rows to {paths.LABEL_REVIEW}")


if __name__ == "__main__":
    main()
