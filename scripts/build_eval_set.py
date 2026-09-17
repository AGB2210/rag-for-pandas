"""Build silver-labelled queries from Stack Overflow questions.

Each question title becomes a query, labelled with the pandas names its
accepted answer uses.

Usage: python scripts/build_eval_set.py
"""

from __future__ import annotations

from rag_for_pandas import paths
from rag_for_pandas.corpus import corpus_owners
from rag_for_pandas.jsonl import load_jsonl, write_jsonl
from rag_for_pandas.labels import build_silver_queries
from rag_for_pandas.stackoverflow import load_items


def main() -> None:
    owners = corpus_owners(load_jsonl(paths.CORPUS))
    questions = load_items("questions_page*.json")
    answers = {a["answer_id"]: a["body"] for a in load_items("answers_page*.json")}

    silver = build_silver_queries(questions, answers, owners)
    paths.SILVER_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.SILVER_QUERIES, silver.records)

    records = silver.records
    print(f"questions with accepted answer: {silver.candidates}")
    print(f"skipped (no usable label):      {silver.skipped_no_label}")
    print(f"eval queries written:           {len(records)}")
    print(f"label word appears in query:    {sum(r['label_in_query'] for r in records)}")
    print("\nmost frequent names across answers:")
    for name, count in silver.frequency.most_common(20):
        print(f"  {count:4d}  ({count / silver.candidates:5.1%})  {name}")


if __name__ == "__main__":
    main()
