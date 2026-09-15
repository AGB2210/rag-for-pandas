"""Fine-tuning the cross-encoder reranker on labelled (question, document) pairs.

Positives are the labelled documents. Negatives are wrong documents from the
retriever's own top results, so the reranker practises on exactly the kind of
list it will later re-order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses.binary_cross_entropy import BinaryCrossEntropyLoss

from docsearch.corpus import document_text, final_name
from docsearch.paths import CHECKPOINTS
from docsearch.training import MAX_LABELS

NEGATIVES_PER_QUESTION = 4
# Candidates are mined from this many retrieved results.
MINING_DEPTH = 20
# Tokens per (question, document) pair. Docstrings start with a summary and the
# parameter list, so the first 256 tokens carry most of what distinguishes them.
MAX_SEQ_LENGTH = 256

EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1


def reranker_pairs(
    docs: list[dict],
    preferred: dict[str, dict],
    queries: list[dict],
    rankings: list[np.ndarray],
    skip_top: int,
    negatives_per_question: int = NEGATIVES_PER_QUESTION,
) -> list[dict]:
    """Labelled pairs: 1 for each labelled document, 0 for retrieved documents that carry no labelled name.

    `skip_top` ignores the first retrieved results when choosing negatives,
    because loose labels miss correct answers and those are likeliest near the top.
    """
    pairs = []
    for query, ranking in zip(queries, rankings):
        if len(query["labels"]) > MAX_LABELS:
            continue
        labels = set(query["labels"])
        for label in query["labels"]:
            pairs.append({"query": query["query"], "response": document_text(preferred[label]), "label": 1})

        negatives = [i for i in ranking[skip_top:MINING_DEPTH] if final_name(docs[i]["qualname"]) not in labels]
        for i in negatives[:negatives_per_question]:
            pairs.append({"query": query["query"], "response": document_text(docs[i]), "label": 0})
    return pairs


def fit_reranker(model: CrossEncoder, pairs: list[dict], seed: int, checkpoints: Path = CHECKPOINTS) -> None:
    """Train in place with binary cross-entropy on relevance labels."""
    model.max_seq_length = MAX_SEQ_LENGTH
    args = CrossEncoderTrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_FRACTION,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
