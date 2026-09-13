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

# A method call whose chain starts at one of these belongs to another library,
# e.g. `np.datetime64(x).astype(datetime)` is NumPy's astype, not pandas'.
FOREIGN_ROOTS: set[str] = {"np", "numpy", "os", "math", "datetime", "re", "json", "time"}
# Documented in pandas, but in answers they are almost always Python list or
# string methods (`rows.append(...)`, `"{}".format(...)`). Found in manual review.
PYTHON_IN_PRACTICE: set[str] = {"append", "clear", "format", "remove", "sort"}
STRING_LITERAL = "<str>"
OPENING = {")": "(", "]": "["}


def load_items(pattern: str) -> list[dict]:
    items: list[dict] = []
    for path in sorted(RAW_DIR.glob(pattern)):
        items.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    return items


def corpus_owners() -> dict[str, set[str]]:
    """Map each final name to the classes documenting it: 'lower' -> {'StringMethods'}."""
    owners: dict[str, set[str]] = {}
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            owner, _, name = json.loads(line)["qualname"].rpartition(".")
            owners.setdefault(name, set()).add(owner)
    return owners


def chain_before(code: str, dot: int) -> list[str]:
    """Names of the expression ending just before `code[dot]`, nearest first.

    For `df["a"].str.lower()` and the dot before `lower`, returns ['str', 'df'].
    Brackets are skipped, so calls and indexers do not break the chain.
    """
    segments: list[str] = []
    i = dot - 1
    while True:
        while i >= 0 and code[i] in OPENING:
            depth, closer = 0, code[i]
            while i >= 0:
                if code[i] == closer:
                    depth += 1
                elif code[i] == OPENING[closer]:
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            i -= 1
        if i >= 0 and code[i] in "'\"":
            segments.append(STRING_LITERAL)
            return segments
        end = i + 1
        while i >= 0 and (code[i].isalnum() or code[i] == "_"):
            i -= 1
        segments.append(code[i + 1 : end])
        if i < 0 or code[i] != ".":
            return segments
        i -= 1


def is_pandas_call(name: str, chain: list[str], owners: dict[str, set[str]]) -> bool:
    if chain[-1] in FOREIGN_ROOTS or chain[-1] == STRING_LITERAL:
        return False
    if name in PYTHON_IN_PRACTICE:
        return False
    # Names pandas only offers through the `.str` accessor: `c.lower()` on a
    # plain Python string is not pandas, `df["c"].str.lower()` is.
    if owners.get(name) == {"StringMethods"}:
        return chain[0] == "str"
    return True


def names_in_answer(body: str, owners: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    for span in CODE_SPAN.findall(body):
        code = html.unescape(span)
        for match in ATTRIBUTE_USE.finditer(code):
            name = match.group(1)
            if is_pandas_call(name, chain_before(code, match.start()), owners):
                found.add(name)
        found.update(QUALIFIED_MENTION.findall(code))
    return found & owners.keys()


def main() -> None:
    owners = corpus_owners()
    # Vote order shifts between fetches, so a question can appear on two pages.
    questions = list({q["question_id"]: q for q in load_items("questions_page*.json")}.values())
    answers = {a["answer_id"]: a["body"] for a in load_items("answers_page*.json")}

    candidates: list[tuple[dict, set[str]]] = []
    for question in questions:
        body = answers.get(question.get("accepted_answer_id"))
        if body is None:
            continue
        candidates.append((question, names_in_answer(body, owners)))

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
