"""Score cited answer generation on the gold set, with automatic checks only.

Retrieval uses the hard-negative retriever; generation uses the prompt in
rag_for_pandas.generation and the local model. For answerable questions, checks
whether answers cite anything, cite numbers that point at no excerpt, cite a
document carrying a correct name, or wrongly abstain. The same checks are
repeated for questions whose excerpts do contain a correct document, which
separates generation failures from retrieval failures. For unanswerable
questions, checks whether the answer abstains.

Also measures whether the answer names a correct function at all, cited or
not. That measure was first computed by hand after reading answers
(experiment 12 in docs/EXPERIMENTS.md) and is now part of the script.

Every answer is saved, so the summary can be printed again without
regenerating (about 18 minutes on a 4 GB GPU) and answers can be read by hand.

Usage:
  python scripts/evaluate_generation.py              generate, save and summarise
  python scripts/evaluate_generation.py --from-saved summarise the saved answers
"""

from __future__ import annotations

import argparse
import time

from rag_for_pandas import paths
from rag_for_pandas.corpus import final_name
from rag_for_pandas.generation import (
    CONTEXT_DOCS,
    build_messages,
    cited_indices,
    cites_correct_document,
    invalid_citations,
    is_abstention,
    names_correct_function,
)
from rag_for_pandas.gold import read_gold_rows
from rag_for_pandas.jsonl import load_jsonl, write_jsonl


def generate_records() -> list[dict]:
    """Retrieve, generate and grade one answer per gold question."""
    # Imported here so --from-saved does not load PyTorch models.
    from rag_for_pandas.local_generator import LocalGenerator
    from rag_for_pandas.retrieval import embedding_rankings

    docs = load_jsonl(paths.CORPUS)
    rows = read_gold_rows()
    rankings = embedding_rankings(docs, [{"query": r["query"]} for r in rows], CONTEXT_DOCS, str(paths.HARD_NEGATIVE_MODEL), cache=None)
    generator = LocalGenerator()

    records = []
    for row, ranking in zip(rows, rankings):
        context = [docs[i] for i in ranking]
        answer = generator.generate(build_messages(row["query"], context))
        labels = [row["primary"], *row["also_correct"].split()] if row["answerable"] == "yes" else []
        records.append({
            "question_id": row["question_id"],
            "query": row["query"],
            "answerable": row["answerable"] == "yes",
            "labels": labels,
            "context": [d["qualname"] for d in context],
            "context_has_correct": any(final_name(d["qualname"]) in labels for d in context),
            "answer": answer,
            "cited": [context[i]["qualname"] for i in cited_indices(answer, len(context))],
            "invalid_citations": invalid_citations(answer, len(context)),
            "cites_correct": cites_correct_document(answer, context, labels),
            "names_correct": names_correct_function(answer, labels),
            "abstained": is_abstention(answer),
        })
    return records


def rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "0/0"


def summary_lines(records: list[dict]) -> list[str]:
    answerable = [r for r in records if r["answerable"]]
    retrieved = [r for r in answerable if r["context_has_correct"]]
    missed = [r for r in answerable if not r["context_has_correct"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def count(items: list[dict], key: str) -> int:
        return sum(bool(r[key]) for r in items)

    return [
        f"gold questions: {len(records)}",
        "",
        f"answerable ({len(answerable)})",
        f"  retriever put a correct document in the {CONTEXT_DOCS} excerpts: {rate(len(retrieved), len(answerable))}",
        f"  answer cites an excerpt:                {rate(count(answerable, 'cited'), len(answerable))}",
        f"  answer cites a number with no excerpt:  {rate(count(answerable, 'invalid_citations'), len(answerable))}",
        f"  answer cites a correct document:        {rate(count(answerable, 'cites_correct'), len(answerable))}",
        f"  answer names a correct function:        {rate(count(answerable, 'names_correct'), len(answerable))}",
        f"  answer wrongly abstains:                {rate(count(answerable, 'abstained'), len(answerable))}",
        "",
        f"answerable, correct document retrieved ({len(retrieved)})",
        f"  answer cites a correct document:        {rate(count(retrieved, 'cites_correct'), len(retrieved))}",
        f"  answer names a correct function:        {rate(count(retrieved, 'names_correct'), len(retrieved))}",
        f"  answer cites an excerpt:                {rate(count(retrieved, 'cited'), len(retrieved))}",
        f"  answer wrongly abstains:                {rate(count(retrieved, 'abstained'), len(retrieved))}",
        "",
        f"answerable, no correct document retrieved ({len(missed)})",
        f"  answer abstains:                        {rate(count(missed, 'abstained'), len(missed))}",
        f"  answer names a correct function anyway: {rate(count(missed, 'names_correct'), len(missed))}",
        "",
        f"unanswerable ({len(unanswerable)})",
        f"  answer abstains:                        {rate(count(unanswerable, 'abstained'), len(unanswerable))}",
        f"  answer cites an excerpt anyway:         {rate(count(unanswerable, 'cited'), len(unanswerable))}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-saved", action="store_true", help=f"summarise {paths.GOLD_ANSWERS} instead of generating")
    args = parser.parse_args()

    if args.from_saved:
        records = load_jsonl(paths.GOLD_ANSWERS)
    else:
        start = time.perf_counter()
        records = generate_records()
        paths.GOLD_ANSWERS.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(paths.GOLD_ANSWERS, records)
        print(f"generated {len(records)} answers in {(time.perf_counter() - start) / 60:.1f} min, saved to {paths.GOLD_ANSWERS}")

    print("\n".join(summary_lines(records)))


if __name__ == "__main__":
    main()
