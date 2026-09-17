"""Split silver queries into train and validation sets with every gold question removed.

Usage: python scripts/split_eval_data.py
"""

from __future__ import annotations

from rag_for_pandas import paths
from rag_for_pandas.gold import read_gold_rows
from rag_for_pandas.jsonl import load_jsonl, write_jsonl
from rag_for_pandas.splits import split_silver


def main() -> None:
    silver = load_jsonl(paths.SILVER_QUERIES)
    gold_rows = read_gold_rows()
    split = split_silver(silver, gold_rows)

    write_jsonl(paths.TRAIN_QUERIES, split.train)
    write_jsonl(paths.VALIDATION_QUERIES, split.validation)

    print(f"silver queries:           {len(silver)}")
    print(f"removed (also in gold):   {split.removed}")
    print(f"train:                    {len(split.train)}  -> {paths.TRAIN_QUERIES}")
    print(f"validation:               {len(split.validation)}  -> {paths.VALIDATION_QUERIES}")
    print(f"gold (test only):         {len(gold_rows)}")


if __name__ == "__main__":
    main()
