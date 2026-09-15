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
  C  B with two worked examples first; adopted as docsearch.generation.build_messages

Usage: python scripts/compare_prompts.py [--questions N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from docsearch import paths
from docsearch.corpus import final_name
from docsearch.generation import (
    CONTEXT_DOCS,
    NOT_FOUND,
    SYSTEM_PROMPT,
    build_messages,
    cited_indices,
    cites_correct_document,
    invalid_citations,
    is_abstention,
    numbered_context,
    user_message,
)
from docsearch.jsonl import load_jsonl
from docsearch.local_generator import LocalGenerator
from docsearch.retrieval import embedding_rankings

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


PROMPTS: dict[str, Callable[[str, list[dict]], list[dict[str, str]]]] = {
    "A rules first": rules_first_messages,
    "B rules last": rules_last_messages,
    "C rules last + examples": build_messages,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=40)
    args = parser.parse_args()

    docs = load_jsonl(paths.CORPUS)
    queries = load_jsonl(paths.VALIDATION_QUERIES)[: args.questions]
    rankings = embedding_rankings(docs, queries, CONTEXT_DOCS, str(paths.HARD_NEGATIVE_MODEL), cache=None)
    contexts = [[docs[i] for i in ranking] for ranking in rankings]
    has_answer = [any(final_name(d["qualname"]) in q["labels"] for d in ctx) for q, ctx in zip(queries, contexts)]
    generator = LocalGenerator()

    answerable, unanswerable = sum(has_answer), len(queries) - sum(has_answer)
    print(f"validation questions: {len(queries)}  (label in retrieved context: {answerable}, not: {unanswerable})")
    print(f"  {'prompt':<26}{'cites':>8}{'invalid':>9}{'correct':>9}{'abstain|answer in ctx':>23}{'abstain|not in ctx':>20}{'s/answer':>10}")

    for name, build in PROMPTS.items():
        start = time.perf_counter()
        answers = [generator.generate(build(q["query"], ctx)) for q, ctx in zip(queries, contexts)]
        seconds = (time.perf_counter() - start) / len(queries)

        cites = sum(bool(cited_indices(a, CONTEXT_DOCS)) for a in answers)
        invalid = sum(bool(invalid_citations(a, CONTEXT_DOCS)) for a in answers)
        correct = sum(cites_correct_document(a, ctx, q["labels"]) for a, ctx, q in zip(answers, contexts, queries))
        wrong_abstain = sum(is_abstention(a) for a, h in zip(answers, has_answer) if h)
        right_abstain = sum(is_abstention(a) for a, h in zip(answers, has_answer) if not h)
        print(
            f"  {name:<26}{cites:>5}/{len(queries):<2}{invalid:>6}/{len(queries):<2}{correct:>6}/{len(queries):<2}"
            f"{wrong_abstain:>19}/{answerable:<3}{right_abstain:>16}/{unanswerable:<3}{seconds:>10.1f}"
        )
        print("RESULT " + json.dumps({
            "prompt": name, "questions": len(queries), "cites": cites, "invalid": invalid, "correct": correct,
            "abstain_when_answer_in_context": wrong_abstain, "abstain_when_not": right_abstain,
            "answers": [{"question": q["query"], "answer": a} for q, a in zip(queries[:5], answers[:5])],
        }), flush=True)


if __name__ == "__main__":
    main()
