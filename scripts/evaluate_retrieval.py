"""Compare BM25, base embeddings and fine-tuned retrievers on silver validation and gold queries.

Usage: python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

from rag_for_pandas import paths
from rag_for_pandas.evaluation import K_VALUES, report_lines
from rag_for_pandas.gold import gold_queries, read_gold_rows
from rag_for_pandas.jsonl import load_jsonl
from rag_for_pandas.retrieval import bm25_rankings, embedding_rankings


def main() -> None:
    docs = load_jsonl(paths.CORPUS)
    depth = max(K_VALUES)
    # Only the validation split: the fine-tuned models have seen the train split,
    # so scoring them on all silver queries would reward memorisation.
    silver = load_jsonl(paths.VALIDATION_QUERIES)
    gold_lenient, gold_strict = gold_queries(read_gold_rows())

    for title, queries in [
        ("SILVER VALIDATION: automatic labels, held out from training", silver),
        ("GOLD: any correct name", gold_lenient),
        ("GOLD: primary name only", gold_strict),
    ]:
        methods = {
            "BM25": bm25_rankings(docs, queries, depth),
            "Embeddings": embedding_rankings(docs, queries, depth),
        }
        if paths.FINETUNED_MODEL.exists():
            methods["Fine-tuned"] = embedding_rankings(docs, queries, depth, str(paths.FINETUNED_MODEL), cache=None)
        if paths.HARD_NEGATIVE_MODEL.exists():
            methods["Fine-tuned+HN"] = embedding_rankings(
                docs, queries, depth, str(paths.HARD_NEGATIVE_MODEL), cache=None
            )
        print("\n".join(report_lines(title, docs, queries, methods)))


if __name__ == "__main__":
    main()
