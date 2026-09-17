"""The pipeline behind the web service: retrieve documentation, then optionally write a cited answer."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol

from rag_for_pandas import paths
from rag_for_pandas.generation import CONTEXT_DOCS, Generator, build_messages, cited_indices, excerpt, invalid_citations, is_abstention
from rag_for_pandas.jsonl import load_jsonl

# Set to a model folder or Hugging Face name; "none" for RAG_FOR_PANDAS_GENERATOR disables answers.
RETRIEVER_ENV = "RAG_FOR_PANDAS_RETRIEVER"
GENERATOR_ENV = "RAG_FOR_PANDAS_GENERATOR"
DISABLED = "none"


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[tuple[int, float]]: ...


@dataclass
class SearchHit:
    rank: int
    name: str
    score: float
    excerpt: str
    source_file: str


@dataclass
class Source:
    number: int  # the [n] an answer uses to cite this document
    name: str
    cited: bool


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    sources: list[Source]
    invalid_citations: list[int]


class Pipeline:
    def __init__(self, docs: list[dict], retriever: Searcher, generator: Generator | None) -> None:
        self.docs = docs
        self.retriever = retriever
        self.generator = generator
        # One GPU cannot safely run two generations at once, so requests take turns.
        # Searches do not take the lock: they are fast and do not wait behind an answer.
        self._generation_lock = threading.Lock()

    @property
    def can_answer(self) -> bool:
        return self.generator is not None

    def search(self, query: str, k: int) -> list[SearchHit]:
        return [
            SearchHit(rank, self.docs[i]["qualname"], round(score, 4), excerpt(self.docs[i]), self.docs[i]["source_file"])
            for rank, (i, score) in enumerate(self.retriever.search(query, k), start=1)
        ]

    def answer(self, question: str) -> AnswerResult:
        if self.generator is None:
            raise RuntimeError("answer generation is disabled")
        context = [self.docs[i] for i, _ in self.retriever.search(question, CONTEXT_DOCS)]
        with self._generation_lock:
            text = self.generator.generate(build_messages(question, context))
        cited = set(cited_indices(text, len(context)))
        return AnswerResult(
            answer=text,
            abstained=is_abstention(text),
            sources=[Source(number + 1, doc["qualname"], number in cited) for number, doc in enumerate(context)],
            invalid_citations=invalid_citations(text, len(context)),
        )


def load_pipeline() -> Pipeline:
    """Load the corpus and models named by environment variables, defaulting to the trained ones."""
    # Imported here so that importing this module, as the tests do, loads no models.
    from sentence_transformers import SentenceTransformer

    from rag_for_pandas.local_generator import LOCAL_MODEL, LocalGenerator
    from rag_for_pandas.retrieval import EmbeddingRetriever

    docs = load_jsonl(paths.CORPUS)
    retriever = EmbeddingRetriever(docs, SentenceTransformer(os.environ.get(RETRIEVER_ENV, str(paths.HARD_NEGATIVE_MODEL))))
    generator_name = os.environ.get(GENERATOR_ENV, LOCAL_MODEL)
    generator = None if generator_name.lower() == DISABLED else LocalGenerator(generator_name)
    return Pipeline(docs, retriever, generator)
