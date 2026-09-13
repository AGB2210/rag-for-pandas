"""Draw a random sample of evaluation labels for manual review.

Writes a CSV with an empty verdict column. The share of wrong labels in a
reviewed sample bounds how far any score measured against them can be trusted.
"""

from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path

from build_eval_set import CODE_SPAN, OUTPUT_PATH as QUERIES_PATH, load_items

REVIEW_PATH = Path("data/eval/label_review.csv")
SAMPLE_SIZE = 50
SEED = 42
PRINT_FIRST = 10


def evidence_lines(answer_body: str, labels: list[str], max_lines: int = 3) -> list[str]:
    """Code lines from the accepted answer that mention any of the labels."""
    lines = []
    for span in CODE_SPAN.findall(answer_body):
        for line in html.unescape(span).splitlines():
            if any(label in line for label in labels):
                lines.append(line.strip()[:100])
    return list(dict.fromkeys(lines))[:max_lines]


def main() -> None:
    with QUERIES_PATH.open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle]
    accepted = {q["question_id"]: q.get("accepted_answer_id") for q in load_items("questions_page*.json")}
    answers = {a["answer_id"]: a["body"] for a in load_items("answers_batch*.json")}

    sample = random.Random(SEED).sample(queries, SAMPLE_SIZE)

    # utf-8-sig adds a byte-order mark so Excel on Windows opens the file with
    # the correct encoding instead of garbling non-ASCII characters.
    with REVIEW_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "verdict", "query", "labels", "evidence", "url", "notes"])
        for review_id, record in enumerate(sample, start=1):
            evidence = evidence_lines(answers[accepted[record["question_id"]]], record["labels"])
            writer.writerow(
                [review_id, "", record["query"], " ".join(record["labels"]),
                 " | ".join(evidence), record["url"], ""]
            )
            if review_id <= PRINT_FIRST:
                print(f"[{review_id}] {record['query']}")
                print(f"     label:    {', '.join(record['labels'])}")
                for line in evidence:
                    print(f"     evidence: {line}")
                print()

    print(f"wrote {SAMPLE_SIZE} rows to {REVIEW_PATH}")


if __name__ == "__main__":
    main()
