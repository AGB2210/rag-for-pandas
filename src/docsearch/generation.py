"""Answer generation: turn retrieved documentation into a short answer with citations.

The generator is an interface, so a local model and an API model are
interchangeable. Answers must cite the excerpts they use as [1], [2], ... and
must say so when the excerpts do not answer the question. Both behaviours can
be checked automatically, without a second model judging the answer.
"""

from __future__ import annotations

import re
from typing import Protocol

from docsearch.corpus import final_name

CONTEXT_DOCS = 3
# The median docstring is 194 words, so most excerpts are complete; long ones are cut.
MAX_EXCERPT_WORDS = 200
NOT_FOUND = "I could not find this in the pandas documentation."

SYSTEM_PROMPT = (
    "You answer questions about pandas using only the numbered documentation excerpts provided. "
    "Name the pandas function that solves the problem and give a short example when the excerpts contain one. "
    "Cite every excerpt you use by its number in square brackets, like [1]. "
    f'If the excerpts do not answer the question, reply exactly: "{NOT_FOUND}"'
)

CITATION = re.compile(r"\[(\d+)\]")
NON_SPACE = re.compile(r"\S+")


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


def excerpt(doc: dict, max_words: int = MAX_EXCERPT_WORDS) -> str:
    """The document's name and docstring, cut after `max_words` words with line breaks kept."""
    text = doc["docstring"]
    words = list(NON_SPACE.finditer(text))
    if len(words) > max_words:
        text = text[: words[max_words - 1].end()] + " ..."
    return f"{doc['qualname']}: {text}"


def build_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    """Chat messages asking for an answer grounded in the numbered excerpts."""
    context = "\n\n".join(f"[{number}] {excerpt(doc)}" for number, doc in enumerate(docs, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


def cited_indices(answer: str, doc_count: int) -> list[int]:
    """0-based indices of the excerpts an answer cites, in order of first citation."""
    cited: list[int] = []
    for match in CITATION.finditer(answer):
        index = int(match.group(1)) - 1
        if 0 <= index < doc_count and index not in cited:
            cited.append(index)
    return cited


def invalid_citations(answer: str, doc_count: int) -> list[int]:
    """Citation numbers that point at no excerpt, such as [4] when only 3 were given."""
    return [int(n) for n in CITATION.findall(answer) if not 1 <= int(n) <= doc_count]


def is_abstention(answer: str) -> bool:
    """Whether the answer says the documentation does not answer the question."""
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def cites_correct_document(answer: str, docs: list[dict], labels: list[str]) -> bool:
    """Whether any cited excerpt is a document carrying a correct name."""
    return any(final_name(docs[i]["qualname"]) in labels for i in cited_indices(answer, len(docs)))
