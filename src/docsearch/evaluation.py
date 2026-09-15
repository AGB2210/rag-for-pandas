"""The comparison table: recall per method and subset, with confidence intervals."""

from __future__ import annotations

import numpy as np

from docsearch.metrics import hits_at_k, paired_bootstrap_ci, recall_at_k

K_VALUES = (1, 5, 10)
# The metric used for every interval, fixed before any comparison was run.
CI_K = 5


def subsets(queries: list[dict]) -> dict[str, list[int]]:
    """All queries, and a split by whether the answer's name already appears in the question."""
    return {
        "ALL QUERIES": list(range(len(queries))),
        "NAME IN TITLE (easy)": [i for i, q in enumerate(queries) if q["label_in_query"]],
        "NAME NOT IN TITLE (honest test)": [i for i, q in enumerate(queries) if not q["label_in_query"]],
    }


def report_lines(title: str, docs: list[dict], queries: list[dict], methods: dict[str, list[np.ndarray]]) -> list[str]:
    lines = ["", f"===== {title} ====="]
    for subset_name, indices in subsets(queries).items():
        subset_queries = [queries[i] for i in indices]
        lines += ["", f"{subset_name}  (n={len(indices)})"]
        lines.append(f"  {'method':<14}" + "".join(f"{'R@' + str(k):>8}" for k in K_VALUES))
        for method_name, rankings in methods.items():
            subset_rankings = [rankings[i] for i in indices]
            values = [recall_at_k(subset_rankings, docs, subset_queries, k) for k in K_VALUES]
            lines.append(f"  {method_name:<14}" + "".join(f"{v:>8.2f}" for v in values))

        names = list(methods)
        # Each method against the one before it, then the final method against the first.
        comparisons = list(zip(names, names[1:]))
        if len(names) > 2:
            comparisons.append((names[0], names[-1]))
        for baseline_name, candidate_name in comparisons:
            baseline, candidate = (
                hits_at_k([methods[name][i] for i in indices], docs, subset_queries, CI_K)
                for name in (baseline_name, candidate_name)
            )
            low, high = paired_bootstrap_ci(baseline, candidate)
            gap = candidate.mean() - baseline.mean()
            lines.append(
                f"  {candidate_name} - {baseline_name} at R@{CI_K}: {gap:+.2f}   95% CI [{low:+.2f}, {high:+.2f}]"
            )
    return lines
