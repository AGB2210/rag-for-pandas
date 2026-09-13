"""Draw the questions for the hand-labelled gold evaluation set.

Samples from every question with an accepted answer, not only those the
automatic labeller could label: questions it skipped are exactly the ones it
handles badly, and leaving them out would make the gold set too easy.

Writes a CSV with empty label columns, filled in by reading each answer:
  primary       the one pandas name that best answers the question
  also_correct  other names that genuinely answer it (space separated)
  answerable    "no" when no documented pandas name answers it
"""

from __future__ import annotations

import csv
import html
import random
from pathlib import Path

from build_eval_set import load_items

GOLD_PATH = Path("annotations/gold_queries.csv")
SAMPLE_SIZE = 100
SEED = 7
COLUMNS = ["question_id", "query", "primary", "also_correct", "answerable", "notes", "url"]


def main() -> None:
    if GOLD_PATH.exists():
        raise SystemExit(f"{GOLD_PATH} exists; refusing to overwrite hand-made labels")

    answered = [q for q in load_items("questions_page*.json") if q.get("accepted_answer_id")]
    answered.sort(key=lambda q: q["question_id"])
    sample = random.Random(SEED).sample(answered, SAMPLE_SIZE)

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for question in sample:
            writer.writerow(
                [question["question_id"], html.unescape(question["title"]), "", "", "", "", question["link"]]
            )
    print(f"wrote {SAMPLE_SIZE} unlabelled rows to {GOLD_PATH}")


if __name__ == "__main__":
    main()
