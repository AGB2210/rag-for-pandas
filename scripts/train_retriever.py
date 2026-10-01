"""Fine-tune the embedding retriever and report validation recall before and after.

Only the train split is learned from and only the validation split is
measured here. The gold set stays untouched until the final evaluation.

Beside the model it saves validation_hits.json: for each validation question,
whether a correct name was in the top 1, 5 and 10. scripts/compare_runs.py
compares runs from those files.

The defaults train the model the service uses. The other options exist for
experiments 19 and 20 in docs/EXPERIMENTS.md:
  --negative-skip N     take hard negatives below rank N instead of 3
  --base-model NAME     start from another embedding model
  --query-prefix TEXT, --document-prefix TEXT
                        text some models expect before every input ("query: " for e5)

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
    MAX_SEQ_LENGTH,
    NEGATIVE_SKIP_TOP,
    SEED,
    Prefixes,
    add_hard_negatives,
    fit,
    preferred_documents,
    training_pairs,
    validation_evaluator,
    validation_hits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hard-negatives", action="store_true", help="add one mined hard negative per pair")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=paths.FINETUNED_MODEL)
    parser.add_argument("--negative-skip", type=int, default=NEGATIVE_SKIP_TOP, help="take hard negatives below this rank")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    args = parser.parse_args()
    prefixes = Prefixes(args.query_prefix, args.document_prefix)

    docs = load_jsonl(paths.CORPUS)
    validation = load_jsonl(paths.VALIDATION_QUERIES)
    preferred = preferred_documents(docs)
    pairs, skipped = training_pairs(preferred, load_jsonl(paths.TRAIN_QUERIES), prefixes)
    evaluator = validation_evaluator(docs, validation, prefixes)
    print(f"training pairs: {len(pairs)}  (questions skipped for > {MAX_LABELS} labels: {skipped})")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = min(model.max_seq_length, MAX_SEQ_LENGTH)
    before = evaluator(model)
    if args.hard_negatives:
        add_hard_negatives(model, pairs, preferred, args.negative_skip, prefixes.document)
    fit(model, pairs, args.hard_negatives, args.seed)

    after = evaluator(model)
    model.save(str(args.output_dir))
    hits = {"question_ids": [q["question_id"] for q in validation], **validation_hits(model, docs, validation, prefixes)}
    (args.output_dir / "validation_hits.json").write_text(json.dumps(hits), encoding="utf-8")

    print(f"\nvalidation, any correct name retrieved  (hard negatives: {args.hard_negatives}, seed: {args.seed})")
    for k in (1, 5, 10):
        key = f"validation_cosine_accuracy@{k}"
        print(f"  R@{k:<3} before {before[key]:.3f}   after {after[key]:.3f}")
    print("RESULT " + json.dumps({
        "base_model": args.base_model,
        "hard_negatives": args.hard_negatives,
        "negative_skip": args.negative_skip if args.hard_negatives else None,
        "seed": args.seed,
        "before_R@5": round(before["validation_cosine_accuracy@5"], 4),
        **{f"R@{k}": round(after[f"validation_cosine_accuracy@{k}"], 4) for k in (1, 5, 10)},
    }))
    print(f"model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
