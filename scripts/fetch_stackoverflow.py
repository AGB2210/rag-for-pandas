"""Fetch pandas-tagged Stack Overflow questions and their accepted answers.

Responses are cached on disk, so re-running only downloads what is missing.

Usage: python scripts/fetch_stackoverflow.py
"""

from __future__ import annotations

from docsearch.stackoverflow import fetch_pages


def main() -> None:
    questions, accepted, answers = fetch_pages()
    print(f"questions fetched:          {questions}")
    print(f"with an accepted answer:    {accepted}")
    print(f"accepted answers fetched:   {answers}")


if __name__ == "__main__":
    main()
