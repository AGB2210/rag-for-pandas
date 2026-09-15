"""File locations shared by every pipeline step, relative to the project root."""

from __future__ import annotations

from pathlib import Path

# Inputs
PANDAS_SOURCE = Path("data/raw/pandas/pandas")
STACKOVERFLOW_RAW = Path("data/raw/stackoverflow")
GOLD_QUERIES = Path("annotations/gold_queries.csv")

# Generated data (not tracked; every file can be rebuilt by a script)
CORPUS = Path("data/interim/docstrings.jsonl")
EMBEDDINGS_CACHE = Path("data/interim/embeddings.npy")
SILVER_QUERIES = Path("data/eval/queries.jsonl")
TRAIN_QUERIES = Path("data/eval/train.jsonl")
VALIDATION_QUERIES = Path("data/eval/validation.jsonl")
LABEL_REVIEW = Path("data/eval/label_review.csv")

# Models
FINETUNED_MODEL = Path("models/retriever-minilm-ft")
HARD_NEGATIVE_MODEL = Path("models/retriever-minilm-ft-hn")
CHECKPOINTS = Path("models/checkpoints")
