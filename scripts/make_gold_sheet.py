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
from itertools import accumulate
from pathlib import Path

from build_eval_set import load_items

GOLD_PATH = Path("annotations/gold_queries.csv")
COLUMNS = ["question_id", "query", "primary", "also_correct", "answerable", "notes", "url"]

# Each round samples only from questions not already in the sheet, and only
# appends, so earlier hand-made labels are never rewritten.
#   round 1: 100 questions from the first 5 fetched pages (500 questions), seed 7
#   round 2: 200 more from all 25 fetched pages (2,500 questions), seed 8
ROUNDS = [(100, 7), (200, 8)]


def main() -> None:
    existing: list[list[str]] = []
    if GOLD_PATH.exists():
        with GOLD_PATH.open(encoding="utf-8") as handle:
            existing = list(csv.reader(handle))[1:]

    # Row counts at which each round starts: [0, 100] for rounds of 100 and 200.
    round_starts = [0, *accumulate(size for size, _ in ROUNDS)][:-1]
    done = len(existing)
    if done not in round_starts:
        raise SystemExit(f"{GOLD_PATH} has {done} rows, which is not the start of a sampling round")
    size, seed = ROUNDS[round_starts.index(done)]

    taken = {int(row[0]) for row in existing}
    # Deduplicate: a question can appear on two pages when votes change between fetches.
    pool = {q["question_id"]: q for q in load_items("questions_page*.json") if q.get("accepted_answer_id")}
    candidates = sorted((q for qid, q in pool.items() if qid not in taken), key=lambda q: q["question_id"])
    sample = random.Random(seed).sample(candidates, size)

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(existing)
        for question in sample:
            writer.writerow(
                [question["question_id"], html.unescape(question["title"]), "", "", "", "", question["link"]]
            )
    print(f"kept {done} labelled rows, appended {size} unlabelled rows to {GOLD_PATH}")


if __name__ == "__main__":
    main()
