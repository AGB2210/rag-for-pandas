"""Compare groups of trained retrievers on the validation questions.

Each run folder holds the validation_hits.json that train_retriever.py saves.
A group is one setting trained with several seeds. For each group this prints
every run's recall and the mean over its runs; for each group after the first
it prints the gap in mean R@5 against the first group, with a paired bootstrap
interval over questions. Hits are averaged over a group's runs first, so the
interval reflects which questions were asked, not which seed was drawn.

Only validation data is read here: settings are never chosen on gold.

Usage:
  python scripts/compare_runs.py --group "skip 3" DIR DIR DIR --group "skip 1" DIR DIR DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag_for_pandas.metrics import paired_bootstrap_ci

DEPTHS = ("R@1", "R@5", "R@10")
DECISION_DEPTH = "R@5"


def load_hits(run: Path) -> dict:
    return json.loads((run / "validation_hits.json").read_text(encoding="utf-8"))


def mean_hits(runs: list[dict], depth: str) -> np.ndarray:
    """Per question, the share of the group's runs that found a correct name."""
    return np.mean([run[depth] for run in runs], axis=0)


def summary_lines(groups: list[tuple[str, list[dict]]]) -> list[str]:
    question_ids = groups[0][1][0]["question_ids"]
    for name, runs in groups:
        if any(run["question_ids"] != question_ids for run in runs):
            raise ValueError(f"runs in '{name}' were scored on different questions")

    lines = [f"validation questions: {len(question_ids)}", f"  {'group':<28}{'runs':>5}" + "".join(f"{depth:>8}" for depth in DEPTHS) + "   R@5 per run"]
    for name, runs in groups:
        means = "".join(f"{mean_hits(runs, depth).mean():>8.3f}" for depth in DEPTHS)
        per_run = ", ".join(f"{np.mean(run[DECISION_DEPTH]):.3f}" for run in runs)
        lines.append(f"  {name:<28}{len(runs):>5}{means}   {per_run}")

    baseline_name, baseline_runs = groups[0]
    baseline = mean_hits(baseline_runs, DECISION_DEPTH)
    for name, runs in groups[1:]:
        candidate = mean_hits(runs, DECISION_DEPTH)
        low, high = paired_bootstrap_ci(baseline, candidate)
        lines.append(f"  {name} - {baseline_name} at {DECISION_DEPTH}: {candidate.mean() - baseline.mean():+.3f}   95% CI [{low:+.3f}, {high:+.3f}]")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "DIR"), help="a name, then its run folders")
    args = parser.parse_args()

    groups = [(name, [load_hits(Path(run)) for run in runs]) for name, *runs in args.group]
    print("\n".join(summary_lines(groups)))


if __name__ == "__main__":
    main()
