"""Fine-tune the embedding retriever on Stack Overflow question/document pairs.

Each training example is (question title, documentation text). The loss pulls
each question towards its document and pushes it away from the other
documents in the same batch, which act as wrong answers.

Only the train split is learned from and only the validation split is
measured here. The gold set stays untouched until the final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

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

from evaluate_retrieval import CORPUS_PATH, MODEL_NAME, document_text, load_jsonl

TRAIN_PATH = Path("data/eval/train.jsonl")
VALIDATION_PATH = Path("data/eval/validation.jsonl")
OUTPUT_DIR = Path("models/retriever-minilm-ft")
CHECKPOINT_DIR = Path("models/checkpoints")

# Questions whose answers call many functions have mostly incidental labels;
# the label review found half of all labels loose.
MAX_LABELS = 3
# When a name exists on several classes, train on the one users call most.
# "" is a top-level function such as read_csv or concat.
OWNER_PREFERENCE = ("", "DataFrame", "NDFrame", "Series", "Index", "DataFrameGroupBy", "SeriesGroupBy", "GroupBy")

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1
SEED = 42


def owner_rank(qualname: str) -> int:
    owner = qualname.rpartition(".")[0]
    return OWNER_PREFERENCE.index(owner) if owner in OWNER_PREFERENCE else len(OWNER_PREFERENCE)


def training_pairs(docs: list[dict], queries: list[dict]) -> tuple[list[dict], int]:
    """One (question, document) pair per label, choosing a single document per name.

    Using one document per name means two correct documents with the same
    name (DataFrame.sum, Series.sum) never meet in a batch as each other's
    wrong answers.
    """
    preferred: dict[str, dict] = {}
    for doc in sorted(docs, key=lambda d: (owner_rank(d["qualname"]), d["qualname"])):
        preferred.setdefault(doc["qualname"].split(".")[-1], doc)

    pairs, skipped = [], 0
    for query in queries:
        if len(query["labels"]) > MAX_LABELS:
            skipped += 1
            continue
        for label in query["labels"]:
            pairs.append({"anchor": query["query"], "positive": document_text(preferred[label])})
    return pairs, skipped


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
    docs = load_jsonl(CORPUS_PATH)
    pairs, skipped = training_pairs(docs, load_jsonl(TRAIN_PATH))
    evaluator = validation_evaluator(docs, load_jsonl(VALIDATION_PATH))
    print(f"training pairs: {len(pairs)}  (questions skipped for > {MAX_LABELS} labels: {skipped})")

    model = SentenceTransformer(MODEL_NAME)
    before = evaluator(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_FRACTION,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=SEED,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        loss=MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
    )
    trainer.train()

    after = evaluator(model)
    model.save(str(OUTPUT_DIR))

    print("\nvalidation, any correct name retrieved")
    for k in (1, 5, 10):
        key = f"validation_cosine_accuracy@{k}"
        print(f"  R@{k:<3} before {before[key]:.3f}   after {after[key]:.3f}")
    print(f"model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
