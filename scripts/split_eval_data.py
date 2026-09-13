"""Split silver-labelled queries into train and validation sets, keeping gold out.

  train       what the retriever learns from
  validation  used while training to choose settings (epochs, learning rate)
  gold        never touched until the final score

A question in both training data and the test set would let the model
memorise the answer, so every gold question is removed first. Titles are also
compared after normalising case and punctuation, because the same question
can be asked twice under different ids.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

SILVER_PATH = Path("data/eval/queries.jsonl")
GOLD_PATH = Path("annotations/gold_queries.csv")
TRAIN_PATH = Path("data/eval/train.jsonl")
VALIDATION_PATH = Path("data/eval/validation.jsonl")
VALIDATION_FRACTION = 0.2
SEED = 13


def normalise_title(title: str) -> str:
    return " ".join(re.findall(r"\w+", title.lower()))


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    with SILVER_PATH.open(encoding="utf-8") as handle:
        silver = [json.loads(line) for line in handle]
    with GOLD_PATH.open(encoding="utf-8") as handle:
        gold = list(csv.DictReader(handle))

    gold_ids = {int(row["question_id"]) for row in gold}
    gold_titles = {normalise_title(row["query"]) for row in gold}

    kept = [
        q for q in silver
        if q["question_id"] not in gold_ids and normalise_title(q["query"]) not in gold_titles
    ]
    removed = len(silver) - len(kept)

    kept.sort(key=lambda q: q["question_id"])
    random.Random(SEED).shuffle(kept)
    cut = round(len(kept) * VALIDATION_FRACTION)
    validation, train = kept[:cut], kept[cut:]

    # The whole point of this script: fail loudly if anything leaks.
    train_ids = {q["question_id"] for q in train}
    validation_ids = {q["question_id"] for q in validation}
    assert not train_ids & validation_ids, "train and validation overlap"
    assert not (train_ids | validation_ids) & gold_ids, "gold question leaked into training data"

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VALIDATION_PATH, validation)

    print(f"silver queries:           {len(silver)}")
    print(f"removed (also in gold):   {removed}")
    print(f"train:                    {len(train)}  -> {TRAIN_PATH}")
    print(f"validation:               {len(validation)}  -> {VALIDATION_PATH}")
    print(f"gold (test only):         {len(gold)}")


if __name__ == "__main__":
    main()
