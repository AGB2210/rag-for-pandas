"""The hand-labelled gold test set: loading it and sampling new rounds of questions.

See annotations/README.md for how rows were labelled.
"""

from __future__ import annotations

import csv
import random
from itertools import accumulate
from pathlib import Path

from rag_for_pandas.paths import GOLD_QUERIES
from rag_for_pandas.text import word_set

COLUMNS = ["question_id", "query", "primary", "also_correct", "answerable", "notes", "url"]

# (size, seed) of each sampling round. Rounds only ever append, so earlier
# hand-made labels are never rewritten.
#   round 1: 100 questions from the first 5 fetched pages (500 questions)
#   round 2: 200 more from all 25 fetched pages (2,500 questions)
ROUNDS = [(100, 7), (200, 8)]


def read_gold_rows(path: Path = GOLD_QUERIES) -> list[dict]:
    # newline="" lets the csv module handle line endings, as its documentation requires.
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gold_queries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Answerable rows as lenient (any correct name) and strict (primary name only) queries.

    Unanswerable rows are left out: no document can be retrieved for them, so
    they would only lower every method's score by the same amount.
    """
    lenient, strict = [], []
    for row in rows:
        if row["answerable"] != "yes":
            continue
        base = {"query": row["query"], "label_in_query": row["primary"].lower() in word_set(row["query"])}
        lenient.append({**base, "labels": [row["primary"], *row["also_correct"].split()]})
        strict.append({**base, "labels": [row["primary"]]})
    return lenient, strict


def next_round(rows_done: int, rounds: list[tuple[int, int]] = ROUNDS) -> tuple[int, int]:
    """(size, seed) of the round that starts after `rows_done` labelled rows."""
    # Row counts at which each round starts: [0, 100] for rounds of 100 and 200.
    round_starts = [0, *accumulate(size for size, _ in rounds)][:-1]
    if rows_done not in round_starts:
        raise ValueError(f"{rows_done} rows is not the start of a sampling round")
    return rounds[round_starts.index(rows_done)]


def sample_round(questions: list[dict], taken: set[int], size: int, seed: int) -> list[dict]:
    """Draw `size` answered questions not already in the gold set."""
    # Deduplicate: a question can appear on two pages when votes change between fetches.
    pool = {q["question_id"]: q for q in questions if q.get("accepted_answer_id")}
    candidates = sorted((q for qid, q in pool.items() if qid not in taken), key=lambda q: q["question_id"])
    return random.Random(seed).sample(candidates, size)
