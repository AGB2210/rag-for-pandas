"""Fine-tuning the embedding retriever on (question, document) pairs.

Each training example is a question title and its documentation text,
optionally with a hard negative: a document that looks relevant but is wrong.
The loss pulls each question towards its document and pushes it away from
the negatives, including the other documents in the same batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import (
    MultipleNegativesRankingLoss,
)

from rag_for_pandas.corpus import document_text, final_name
from rag_for_pandas.paths import CHECKPOINTS
from rag_for_pandas.retrieval import Encoder

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
        preferred.setdefault(final_name(doc["qualname"]), doc)
    return preferred


def training_pairs(preferred: dict[str, dict], queries: list[dict]) -> tuple[list[dict], int]:
    """One (question, document) pair per label, skipping questions with too many labels.

    Returns the pairs and the number of questions skipped.
    """
    pairs, skipped = [], 0
    for query in queries:
        if len(query["labels"]) > MAX_LABELS:
            skipped += 1
            continue
        for label in query["labels"]:
            pairs.append({"anchor": query["query"], "positive": document_text(preferred[label]), "labels": query["labels"]})
    return pairs, skipped


def add_hard_negatives(encoder: Encoder, pairs: list[dict], preferred: dict[str, dict]) -> None:
    """Give each pair the highest-ranked document below NEGATIVE_SKIP_TOP that is not a labelled answer."""
    names = list(preferred)
    doc_vectors = encoder.encode([document_text(preferred[n]) for n in names], normalize_embeddings=True, batch_size=64)
    anchors = sorted({pair["anchor"] for pair in pairs})
    query_vectors = encoder.encode(anchors, normalize_embeddings=True, batch_size=64)
    rankings = dict(zip(anchors, np.argsort(-(query_vectors @ doc_vectors.T), axis=1)))

    for pair in pairs:
        candidates = rankings[pair["anchor"]][NEGATIVE_SKIP_TOP:]
        negative = next(names[i] for i in candidates if names[i] not in pair["labels"])
        pair["negative"] = document_text(preferred[negative])


def validation_evaluator(docs: list[dict], queries: list[dict]) -> InformationRetrievalEvaluator:
    """Counts a hit when any document carrying a labelled name is retrieved, as the benchmark does."""
    ids_by_name: dict[str, set[str]] = {}
    for index, doc in enumerate(docs):
        ids_by_name.setdefault(final_name(doc["qualname"]), set()).add(str(index))
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


def fit(model: SentenceTransformer, pairs: list[dict], hard_negatives: bool, seed: int, checkpoints: Path = CHECKPOINTS) -> None:
    """Train in place with MultipleNegativesRankingLoss and a no-duplicates batch sampler."""
    columns = ["anchor", "positive", *(["negative"] if hard_negatives else [])]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints),
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
        seed=seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list([{column: pair[column] for column in columns} for pair in pairs]),
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
