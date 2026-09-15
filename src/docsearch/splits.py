"""Split silver-labelled queries into train and validation sets, keeping gold out.

  train       what the retriever learns from
  validation  used while training to choose settings
  gold        never touched until the final score

A question in both training data and the test set would let the model
memorise the answer, so every gold question is removed first. Titles are also
compared after normalising case and punctuation, because the same question
can be asked twice under different ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from docsearch.text import normalise_title

VALIDATION_FRACTION = 0.2
SEED = 13


@dataclass
class Split:
    train: list[dict]
    validation: list[dict]
    removed: int  # silver queries dropped because they are gold questions


class LeakageError(RuntimeError):
    """A test question reached the data a model learns from or is tuned on."""


def split_silver(
    silver: list[dict], gold_rows: list[dict], fraction: float = VALIDATION_FRACTION, seed: int = SEED
) -> Split:
    gold_ids = {int(row["question_id"]) for row in gold_rows}
    gold_titles = {normalise_title(row["query"]) for row in gold_rows}

    kept = [
        q for q in silver
        if q["question_id"] not in gold_ids and normalise_title(q["query"]) not in gold_titles
    ]
    # Sort before shuffling so the split depends only on the seed, not input order.
    kept.sort(key=lambda q: q["question_id"])
    random.Random(seed).shuffle(kept)
    cut = round(len(kept) * fraction)
    validation, train = kept[:cut], kept[cut:]

    # Checked explicitly rather than with assert, which `python -O` removes.
    train_ids = {q["question_id"] for q in train}
    validation_ids = {q["question_id"] for q in validation}
    if train_ids & validation_ids:
        raise LeakageError("train and validation overlap")
    if (train_ids | validation_ids) & gold_ids:
        raise LeakageError("gold question leaked into training data")

    return Split(train, validation, len(silver) - len(kept))
