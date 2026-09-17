"""Small text helpers used by labelling, splitting and search."""

from __future__ import annotations

import re

WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens.

    \\w+ splits "DataFrame.dropna" into "dataframe" and "dropna" so a query
    containing "dropna" can match. A sloppy tokenizer would weaken the BM25
    baseline and exaggerate every later improvement.
    """
    return WORD.findall(text.lower())


def word_set(text: str) -> set[str]:
    return set(tokenize(text))


def normalise_title(title: str) -> str:
    """Case- and punctuation-insensitive form, to catch the same question asked twice."""
    return " ".join(tokenize(title))
