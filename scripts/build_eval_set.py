"""Build a retrieval evaluation set from Stack Overflow questions.

Each question title becomes a test query. Its labels are the pandas API names
used in the accepted answer that also exist in our corpus.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw/stackoverflow")
CORPUS_PATH = Path("data/interim/docstrings.jsonl")
OUTPUT_PATH = Path("data/eval/queries.jsonl")

CODE_SPAN = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# Attribute followed by a call or an indexer. The `[` case is essential:
# indexers such as `.loc` and `.iloc` are never written with parentheses.
ATTRIBUTE_USE = re.compile(r"\.([A-Za-z_]\w*)\s*[(\[]")
# Qualified mentions without a call, e.g. "DataFrame.iterrows" written in prose.
QUALIFIED_MENTION = re.compile(r"\b(?:pd|pandas|DataFrame|Series|Index)\.([A-Za-z_]\w*)")

# Names found in so many answers that they say nothing about what an answer is
# really about. Chosen from the frequency report this script prints.
GENERIC_NAMES: set[str] = {"DataFrame", "Series"}


def load_items(pattern: str) -> list[dict]:
    items: list[dict] = []
    for path in sorted(RAW_DIR.glob(pattern)):
        items.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    return items


def corpus_names() -> set[str]:
    """Final segment of every documented name: 'DataFrame.dropna' -> 'dropna'."""
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        return {json.loads(line)["qualname"].split(".")[-1] for line in handle}


def names_in_answer(body: str, known: set[str]) -> set[str]:
    found: set[str] = set()
    for span in CODE_SPAN.findall(body):
        code = html.unescape(span)
        found.update(ATTRIBUTE_USE.findall(code))
        found.update(QUALIFIED_MENTION.findall(code))
    return found & known


def main() -> None:
    known = corpus_names()
    questions = load_items("questions_page*.json")
    answers = {a["answer_id"]: a["body"] for a in load_items("answers_batch*.json")}

    candidates: list[tuple[dict, set[str]]] = []
    for question in questions:
        body = answers.get(question.get("accepted_answer_id"))
        if body is None:
            continue
        candidates.append((question, names_in_answer(body, known)))

    frequency = Counter(name for _, names in candidates for name in names)

    records = []
    skipped_no_label = 0
    for question, names in candidates:
        labels = sorted(names - GENERIC_NAMES)
        if not labels:
            skipped_no_label += 1
            continue
        query = html.unescape(question["title"])
        query_words = set(re.findall(r"\w+", query.lower()))
        records.append(
            {
                "question_id": question["question_id"],
                "query": query,
                "labels": labels,
                "label_in_query": any(label.lower() in query_words for label in labels),
                "url": question.get("link"),
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"questions with accepted answer: {len(candidates)}")
    print(f"skipped (no usable label):      {skipped_no_label}")
    print(f"eval queries written:           {len(records)}")
    print(f"label word appears in query:    {sum(r['label_in_query'] for r in records)}")
    print("\nmost frequent names across answers:")
    for name, count in frequency.most_common(20):
        print(f"  {count:4d}  ({count / len(candidates):5.1%})  {name}")


if __name__ == "__main__":
    main()
