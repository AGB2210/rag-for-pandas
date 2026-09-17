"""Fine-tune the embedding retriever and report validation recall before and after.

Only the train split is learned from and only the validation split is
measured here. The gold set stays untouched until the final evaluation.

Usage:
  python scripts/train_retriever.py [--hard-negatives] [--seed N] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_for_pandas import paths
from rag_for_pandas.jsonl import load_jsonl
from rag_for_pandas.retrieval import BASE_MODEL
from rag_for_pandas.training import (
    MAX_LABELS,
    SEED,
    add_hard_negatives,
    fit,
    preferred_documents,
    training_pairs,
    validation_evaluator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hard-negatives", action="store_true", help="add one mined hard negative per pair")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=paths.FINETUNED_MODEL)
    args = parser.parse_args()

    docs = load_jsonl(paths.CORPUS)
    preferred = preferred_documents(docs)
    pairs, skipped = training_pairs(preferred, load_jsonl(paths.TRAIN_QUERIES))
    evaluator = validation_evaluator(docs, load_jsonl(paths.VALIDATION_QUERIES))
    print(f"training pairs: {len(pairs)}  (questions skipped for > {MAX_LABELS} labels: {skipped})")

    model = SentenceTransformer(BASE_MODEL)
    before = evaluator(model)
    if args.hard_negatives:
        add_hard_negatives(model, pairs, preferred)
    fit(model, pairs, args.hard_negatives, args.seed)

    after = evaluator(model)
    model.save(str(args.output_dir))

    print(f"\nvalidation, any correct name retrieved  (hard negatives: {args.hard_negatives}, seed: {args.seed})")
    for k in (1, 5, 10):
        key = f"validation_cosine_accuracy@{k}"
        print(f"  R@{k:<3} before {before[key]:.3f}   after {after[key]:.3f}")
    print("RESULT " + json.dumps({
        "hard_negatives": args.hard_negatives,
        "seed": args.seed,
        **{f"R@{k}": round(after[f"validation_cosine_accuracy@{k}"], 4) for k in (1, 5, 10)},
    }))
    print(f"model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
