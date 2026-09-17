"""Compare answer-generation prompts on validation questions, with automatic checks only.

For each prompt, measures how often answers cite an excerpt, cite a number that
points at no excerpt, cite a document carrying a silver label, and abstain.
Abstention is split by whether the retrieved excerpts contain a labelled
document: abstaining is right when they do not, and wrong when they do.
Silver labels are loose, so these rates are approximate; gold is used only
for the final score of the chosen prompt.

Prompts compared (every prompt sees the same retrieved excerpts):
  A  rules in the system message, before the excerpts (the first prompt tried)
  B  shorter, stricter rules placed after the question
  C  B with two worked examples first; adopted as rag_for_pandas.generation.build_messages
  D  C, but the citation goes right after each function named: "DataFrame.dropna [1]"
  E  C, but citations go on a fixed last line: "Sources: [1]"
  F  E, but refuse only when no excerpt is about the question

D and E target the main failure of C: answers that name the right function
without citing its excerpt.

Usage: python scripts/compare_prompts.py [--questions N] [--start N] [--prompts CDEF]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from rag_for_pandas import paths
from rag_for_pandas.corpus import final_name
from rag_for_pandas.generation import (
    CONTEXT_DOCS,
    NOT_FOUND,
    RULES,
    SYSTEM_PROMPT,
    build_messages,
    cited_indices,
    cites_correct_document,
    invalid_citations,
    is_abstention,
    names_correct_function,
    numbered_context,
    user_message,
)
from rag_for_pandas.jsonl import load_jsonl
from rag_for_pandas.local_generator import LocalGenerator
from rag_for_pandas.retrieval import embedding_rankings

FIRST_SYSTEM_PROMPT = (
    "You answer questions about pandas using only the numbered documentation excerpts provided. "
    "Name the pandas function that solves the problem and give a short example when the excerpts contain one. "
    "Cite every excerpt you use by its number in square brackets, like [1]. "
    f'If the excerpts do not answer the question, reply exactly: "{NOT_FOUND}"'
)


def rules_first_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FIRST_SYSTEM_PROMPT},
        {"role": "user", "content": f"Documentation excerpts:\n\n{numbered_context(docs)}\n\nQuestion: {question}"},
    ]


def rules_last_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(question, numbered_context(docs))},
    ]


CITATION_RULE = "- End every sentence that uses an excerpt with its number in square brackets, like [2].\n"
assert CITATION_RULE in RULES

NAME_CITATION_RULES = RULES.replace(
    CITATION_RULE, "- Write the excerpt number right after every pandas function you name, like DataFrame.dropna [1].\n"
)
NAME_CITATION_EXAMPLES = [
    ("How do I delete rows with empty cells?", "Use DataFrame.dropna [1], which removes rows that contain missing values."),
    ("How do I read an Excel file?", NOT_FOUND),
]

SOURCES_LINE_RULES = RULES.replace(
    CITATION_RULE, '- End with a last line "Sources:" followed by the numbers of the excerpts you used, like "Sources: [1]".\n'
)
SOURCES_LINE_EXAMPLES = [
    ("How do I delete rows with empty cells?", "Use DataFrame.dropna, which removes rows that contain missing values.\nSources: [1]"),
    ("How do I read an Excel file?", NOT_FOUND),
]

ABSTAIN_RULE = f'- If no excerpt answers the question, reply exactly: "{NOT_FOUND}"'
assert RULES.endswith(ABSTAIN_RULE)

# E refused in 9 of the 12 questions where it lost a correct name that C gave,
# often with the answer in an excerpt; F narrows when to refuse.
USE_RELATED_RULES = SOURCES_LINE_RULES.replace(
    ABSTAIN_RULE,
    "- If an excerpt describes a function that solves the question, answer with it, "
    "even when the excerpt does not show your exact case.\n"
    f'- If no excerpt is about the question, reply exactly: "{NOT_FOUND}"',
)


def name_citation_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    return build_messages(question, docs, NAME_CITATION_RULES, NAME_CITATION_EXAMPLES)


def sources_line_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    return build_messages(question, docs, SOURCES_LINE_RULES, SOURCES_LINE_EXAMPLES)


def use_related_messages(question: str, docs: list[dict]) -> list[dict[str, str]]:
    return build_messages(question, docs, USE_RELATED_RULES, SOURCES_LINE_EXAMPLES)


PROMPTS: dict[str, Callable[[str, list[dict]], list[dict[str, str]]]] = {
    "A rules first": rules_first_messages,
    "B rules last": rules_last_messages,
    "C rules last + examples": build_messages,
    "D cite after each name": name_citation_messages,
    "E sources line": sources_line_messages,
    "F sources line, refuse less": use_related_messages,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--start", type=int, default=0, help="skip this many validation questions, to test on unseen ones")
    parser.add_argument("--prompts", default="".join(name[0] for name in PROMPTS), help="prompt letters to run, such as CDE")
    args = parser.parse_args()

    docs = load_jsonl(paths.CORPUS)
    queries = load_jsonl(paths.VALIDATION_QUERIES)[args.start : args.start + args.questions]
    rankings = embedding_rankings(docs, queries, CONTEXT_DOCS, str(paths.HARD_NEGATIVE_MODEL), cache=None)
    contexts = [[docs[i] for i in ranking] for ranking in rankings]
    has_answer = [any(final_name(d["qualname"]) in q["labels"] for d in ctx) for q, ctx in zip(queries, contexts)]
    generator = LocalGenerator()

    answerable, unanswerable = sum(has_answer), len(queries) - sum(has_answer)
    print(f"validation questions: {len(queries)}  (label in retrieved context: {answerable}, not: {unanswerable})")
    print(f"  {'prompt':<26}{'cites':>8}{'invalid':>9}{'correct':>9}{'names':>8}{'abstain|answer in ctx':>23}{'abstain|not in ctx':>20}{'s/answer':>10}")

    for name, build in PROMPTS.items():
        if name[0] not in args.prompts:
            continue
        start = time.perf_counter()
        answers = [generator.generate(build(q["query"], ctx)) for q, ctx in zip(queries, contexts)]
        seconds = (time.perf_counter() - start) / len(queries)

        cites = sum(bool(cited_indices(a, CONTEXT_DOCS)) for a in answers)
        invalid = sum(bool(invalid_citations(a, CONTEXT_DOCS)) for a in answers)
        correct = sum(cites_correct_document(a, ctx, q["labels"]) for a, ctx, q in zip(answers, contexts, queries))
        names = sum(names_correct_function(a, q["labels"]) for a, q in zip(answers, queries))
        wrong_abstain = sum(is_abstention(a) for a, h in zip(answers, has_answer) if h)
        right_abstain = sum(is_abstention(a) for a, h in zip(answers, has_answer) if not h)
        print(
            f"  {name:<26}{cites:>5}/{len(queries):<2}{invalid:>6}/{len(queries):<2}{correct:>6}/{len(queries):<2}{names:>5}/{len(queries):<2}"
            f"{wrong_abstain:>19}/{answerable:<3}{right_abstain:>16}/{unanswerable:<3}{seconds:>10.1f}"
        )
        print("RESULT " + json.dumps({
            "prompt": name, "questions": len(queries), "cites": cites, "invalid": invalid, "correct": correct, "names_correct": names,
            "abstain_when_answer_in_context": wrong_abstain, "abstain_when_not": right_abstain,
            "answers": [{"question": q["query"], "answer": a} for q, a in zip(queries, answers)],
        }), flush=True)


if __name__ == "__main__":
    main()
