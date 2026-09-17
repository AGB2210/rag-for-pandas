"""Silver labels: pandas names used in each question's accepted answer.

Automatic and cheap, but loose: a manual review of 50 labels found 34% correct,
50% loose and 16% wrong. Used for training and validation, never as the final test.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass

from rag_for_pandas.text import word_set

CODE_SPAN = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# Attribute followed by a call or an indexer. The `[` case is essential:
# indexers such as `.loc` and `.iloc` are never written with parentheses.
ATTRIBUTE_USE = re.compile(r"\.([A-Za-z_]\w*)\s*[(\[]")
# Qualified mentions without a call, e.g. "DataFrame.iterrows" written in prose.
QUALIFIED_MENTION = re.compile(r"\b(?:pd|pandas|DataFrame|Series|Index)\.([A-Za-z_]\w*)")

# Names found in so many answers that they say nothing about what an answer is
# really about. Chosen from the name frequency report.
GENERIC_NAMES = frozenset({"DataFrame", "Series"})

# A method call whose chain starts at one of these belongs to another library,
# e.g. `np.datetime64(x).astype(datetime)` is NumPy's astype, not pandas'.
FOREIGN_ROOTS = frozenset({"np", "numpy", "os", "math", "datetime", "re", "json", "time"})
# Documented in pandas, but in answers they are almost always Python list or
# string methods (`rows.append(...)`, `"{}".format(...)`). Found in manual review.
PYTHON_IN_PRACTICE = frozenset({"append", "clear", "format", "remove", "sort"})
STRING_LITERAL = "<str>"
OPENING = {")": "(", "]": "["}


@dataclass
class SilverLabels:
    records: list[dict]
    candidates: int  # questions with an accepted answer
    skipped_no_label: int
    frequency: Counter  # how many answers use each name, before removing generic names


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
    """Documented pandas names used in the code of an answer's HTML body."""
    found: set[str] = set()
    for span in CODE_SPAN.findall(body):
        code = html.unescape(span)
        for match in ATTRIBUTE_USE.finditer(code):
            name = match.group(1)
            if is_pandas_call(name, chain_before(code, match.start()), owners):
                found.add(name)
        found.update(QUALIFIED_MENTION.findall(code))
    return found & owners.keys()


def build_silver_queries(questions: list[dict], answers: dict[int, str], owners: dict[str, set[str]]) -> SilverLabels:
    """One labelled query per question whose accepted answer uses a specific pandas name."""
    # Vote order shifts between fetches, so a question can appear on two pages.
    unique_questions = list({q["question_id"]: q for q in questions}.values())

    candidates: list[tuple[dict, set[str]]] = []
    for question in unique_questions:
        body = answers.get(question.get("accepted_answer_id"))
        if body is None:
            continue
        candidates.append((question, names_in_answer(body, owners)))

    records = []
    skipped_no_label = 0
    for question, names in candidates:
        labels = sorted(names - GENERIC_NAMES)
        if not labels:
            skipped_no_label += 1
            continue
        query = html.unescape(question["title"])
        query_words = word_set(query)
        records.append(
            {
                "question_id": question["question_id"],
                "query": query,
                "labels": labels,
                "label_in_query": any(label.lower() in query_words for label in labels),
                "url": question.get("link"),
            }
        )

    frequency = Counter(name for _, names in candidates for name in names)
    return SilverLabels(records, len(candidates), skipped_no_label, frequency)


def evidence_lines(answer_body: str, labels: list[str], max_lines: int = 3) -> list[str]:
    """Code lines from an answer that mention any of the labels, for manual review."""
    lines = []
    for span in CODE_SPAN.findall(answer_body):
        for line in html.unescape(span).splitlines():
            if any(label in line for label in labels):
                lines.append(line.strip()[:100])
    return list(dict.fromkeys(lines))[:max_lines]
