"""Append the next round of unlabelled questions to the gold sheet.

Samples from every answered question, including those the automatic labeller
skipped: they are exactly the ones it handles badly, and leaving them out would
make the gold set too easy. New rows get empty label columns, filled in by
reading each answer. Existing rows are written back unchanged.

Usage: python scripts/make_gold_sheet.py
"""

from __future__ import annotations

import csv
import html

from docsearch.gold import COLUMNS, next_round, sample_round
from docsearch.paths import GOLD_QUERIES
from docsearch.stackoverflow import load_items


def main() -> None:
    existing: list[list[str]] = []
    if GOLD_QUERIES.exists():
        with GOLD_QUERIES.open(encoding="utf-8") as handle:
            existing = list(csv.reader(handle))[1:]

    done = len(existing)
    try:
        size, seed = next_round(done)
    except ValueError:
        raise SystemExit(f"{GOLD_QUERIES} has {done} rows, which is not the start of a sampling round")

    taken = {int(row[0]) for row in existing}
    sample = sample_round(load_items("questions_page*.json"), taken, size, seed)

    GOLD_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_QUERIES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(existing)
        for question in sample:
            writer.writerow([question["question_id"], html.unescape(question["title"]), "", "", "", "", question["link"]])
    print(f"kept {done} labelled rows, appended {size} unlabelled rows to {GOLD_QUERIES}")


if __name__ == "__main__":
    main()
