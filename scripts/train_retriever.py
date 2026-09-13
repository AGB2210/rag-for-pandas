"""Fine-tune the embedding retriever on Stack Overflow question/document pairs.

Each training example is (question title, documentation text), optionally with
a hard negative: a document that looks relevant but is wrong. The loss pulls
each question towards its document and pushes it away from the negatives,
including the other documents in the same batch.

Only the train split is learned from and only the validation split is
measured here. The gold set stays untouched until the final evaluation.

Usage:
  python scripts/train_retriever.py [--hard-negatives] [--seed N] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import (
    MultipleNegativesRankingLoss,
)

from evaluate_retrieval import CORPUS_PATH, FINETUNED_MODEL, MODEL_NAME, document_text, load_jsonl

TRAIN_PATH = Path("data/eval/train.jsonl")
VALIDATION_PATH = Path("data/eval/validation.jsonl")
CHECKPOINT_DIR = Path("models/checkpoints")

# Questions whose answers call many functions have mostly incidental labels;
# the label review found half of all labels loose.
MAX_LABELS = 3
# When a name exists on several classes, train on the one users call most.
# "" is a top-level function such as read_csv or concat.
OWNER_PREFERENCE = ("", "DataFrame", "NDFrame", "Series", "Index", "DataFrameGroupBy", "SeriesGroupBy", "GroupBy")
# Hard negatives are taken below this rank. Silver labels miss correct answers,
# so the documents ranked highest are often right, and pushing them away would
# teach the model the wrong thing.
NEGATIVE_SKIP_TOP = 3

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1
SEED = 42


def owner_rank(qualname: str) -> int:
    owner = qualname.rpartition(".")[0]
    return OWNER_PREFERENCE.index(owner) if owner in OWNER_PREFERENCE else len(OWNER_PREFERENCE)


def preferred_documents(docs: list[dict]) -> dict[str, dict]:
    """One document per name: 'sum' -> DataFrame.sum rather than Series.sum.

    Positives and negatives both come from this mapping, so two correct
    documents with the same name never meet in a batch as each other's
    wrong answers.
    """
    preferred: dict[str, dict] = {}
    for doc in sorted(docs, key=lambda d: (owner_rank(d["qualname"]), d["qualname"])):
        preferred.setdefault(doc["qualname"].split(".")[-1], doc)
    return preferred


def training_pairs(preferred: dict[str, dict], queries: list[dict]) -> tuple[list[dict], int]:
    """One (question, document) pair per label, skipping questions with too many labels."""
    pairs, skipped = [], 0
    for query in queries:
        if len(query["labels"]) > MAX_LABELS:
            skipped += 1
            continue
        for label in query["labels"]:
            pairs.append({"anchor": query["query"], "positive": document_text(preferred[label]), "labels": query["labels"]})
    return pairs, skipped


def add_hard_negatives(model: SentenceTransformer, pairs: list[dict], preferred: dict[str, dict]) -> None:
    """Give each pair the highest-ranked document below NEGATIVE_SKIP_TOP that is not a labelled answer."""
    names = list(preferred)
    doc_vectors = model.encode([document_text(preferred[n]) for n in names], normalize_embeddings=True, batch_size=64)
    anchors = sorted({pair["anchor"] for pair in pairs})
    query_vectors = model.encode(anchors, normalize_embeddings=True, batch_size=64)
    rankings = dict(zip(anchors, np.argsort(-(query_vectors @ doc_vectors.T), axis=1)))

    for pair in pairs:
        candidates = rankings[pair["anchor"]][NEGATIVE_SKIP_TOP:]
        negative = next(names[i] for i in candidates if names[i] not in pair["labels"])
        pair["negative"] = document_text(preferred[negative])


def validation_evaluator(docs: list[dict], queries: list[dict]) -> InformationRetrievalEvaluator:
    """Counts a hit when any document carrying a labelled name is retrieved, as evaluate_retrieval does."""
    ids_by_name: dict[str, set[str]] = {}
    for index, doc in enumerate(docs):
        ids_by_name.setdefault(doc["qualname"].split(".")[-1], set()).add(str(index))
    return InformationRetrievalEvaluator(
        queries={str(q["question_id"]): q["query"] for q in queries},
        corpus={str(i): document_text(d) for i, d in enumerate(docs)},
        relevant_docs={
            str(q["question_id"]): set().union(*(ids_by_name[label] for label in q["labels"]))
            for q in queries
        },
        accuracy_at_k=[1, 5, 10],
        name="validation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hard-negatives", action="store_true", help="add one mined hard negative per pair")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=FINETUNED_MODEL)
    args = parser.parse_args()

    docs = load_jsonl(CORPUS_PATH)
    preferred = preferred_documents(docs)
    pairs, skipped = training_pairs(preferred, load_jsonl(TRAIN_PATH))
    evaluator = validation_evaluator(docs, load_jsonl(VALIDATION_PATH))
    print(f"training pairs: {len(pairs)}  (questions skipped for > {MAX_LABELS} labels: {skipped})")

    model = SentenceTransformer(MODEL_NAME)
    before = evaluator(model)
    columns = ["anchor", "positive"]
    if args.hard_negatives:
        add_hard_negatives(model, pairs, preferred)
        columns.append("negative")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_FRACTION,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list([{column: pair[column] for column in columns} for pair in pairs]),
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()

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
