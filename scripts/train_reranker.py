"""Fine-tune the cross-encoder reranker on training questions.

Negatives are mined from the hard-negative retriever's results for each
training question. Score the saved model with:
  python scripts/tune_reranker.py --model <output-dir>

Usage:
  python scripts/train_reranker.py [--skip-top N] [--seed N] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder

from rag_for_pandas import paths
from rag_for_pandas.jsonl import load_jsonl
from rag_for_pandas.reranker_training import MINING_DEPTH, fit_reranker, reranker_pairs
from rag_for_pandas.reranking import RERANK_MODEL
from rag_for_pandas.retrieval import embedding_rankings
from rag_for_pandas.training import SEED, preferred_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-top", type=int, default=0, help="retrieved results ignored when mining negatives")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=paths.RERANKER_MODEL)
    args = parser.parse_args()

    docs = load_jsonl(paths.CORPUS)
    train = load_jsonl(paths.TRAIN_QUERIES)
    rankings = embedding_rankings(docs, train, MINING_DEPTH, str(paths.HARD_NEGATIVE_MODEL), cache=None)
    pairs = reranker_pairs(docs, preferred_documents(docs), train, rankings, args.skip_top)
    counts = Counter(p["label"] for p in pairs)
    print(f"reranker pairs: {len(pairs)}  (positive {counts[1]}, negative {counts[0]}, skip top {args.skip_top})")

    model = CrossEncoder(RERANK_MODEL)
    fit_reranker(model, pairs, args.seed)
    model.save(str(args.output_dir))
    print(f"model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
