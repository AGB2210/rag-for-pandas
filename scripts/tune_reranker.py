"""Compare reranking depths for a reranker, using validation only.

Compares the hard-negative retriever alone with the retriever followed by a
cross-encoder over its top 10, 20 or 50 results, and reports reranking time.
The gold set is not touched: choices are fixed before gold is scored.

Usage:
  python scripts/tune_reranker.py [--model NAME_OR_PATH]
"""

from __future__ import annotations

import argparse
import json
import time

from sentence_transformers.cross_encoder import CrossEncoder

from docsearch import paths
from docsearch.evaluation import CI_K, K_VALUES
from docsearch.jsonl import load_jsonl
from docsearch.metrics import hits_at_k, paired_bootstrap_ci, recall_at_k
from docsearch.reranker_training import MAX_SEQ_LENGTH
from docsearch.reranking import RERANK_MODEL, rerank
from docsearch.retrieval import embedding_rankings

DEPTHS = (10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=RERANK_MODEL, help="Hugging Face name or local path of a cross-encoder")
    args = parser.parse_args()

    docs = load_jsonl(paths.CORPUS)
    queries = load_jsonl(paths.VALIDATION_QUERIES)
    retrieved = embedding_rankings(docs, queries, max(DEPTHS), str(paths.HARD_NEGATIVE_MODEL), cache=None)
    scorer = CrossEncoder(args.model)
    # Score with the same input length the fine-tuned rerankers were trained with.
    scorer.max_seq_length = MAX_SEQ_LENGTH

    print(f"validation queries: {len(queries)}   reranker: {args.model}")
    print(f"  {'method':<22}" + "".join(f"{'R@' + str(k):>8}" for k in K_VALUES) + f"{'ms/query':>10}   gap at R@{CI_K} vs retriever")
    baseline_hits = hits_at_k(retrieved, docs, queries, CI_K)
    print(f"  {'retriever only':<22}" + "".join(f"{recall_at_k(retrieved, docs, queries, k):>8.2f}" for k in K_VALUES))

    for depth in DEPTHS:
        start = time.perf_counter()
        reranked = rerank(scorer, docs, queries, retrieved, depth)
        ms_per_query = (time.perf_counter() - start) * 1000 / len(queries)

        recalls = {k: recall_at_k(reranked, docs, queries, k) for k in K_VALUES}
        candidate_hits = hits_at_k(reranked, docs, queries, CI_K)
        low, high = paired_bootstrap_ci(baseline_hits, candidate_hits)
        gap = candidate_hits.mean() - baseline_hits.mean()
        print(
            f"  {'rerank top ' + str(depth):<22}"
            + "".join(f"{recalls[k]:>8.2f}" for k in K_VALUES)
            + f"{ms_per_query:>10.1f}   {gap:+.2f} [{low:+.2f}, {high:+.2f}]"
        )
        print("RESULT " + json.dumps({"model": args.model, "depth": depth, **{f"R@{k}": round(v, 4) for k, v in recalls.items()}}))


if __name__ == "__main__":
    main()
