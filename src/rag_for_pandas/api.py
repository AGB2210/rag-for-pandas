"""HTTP API: search the pandas documentation and ask for cited answers.

Run with:
  uvicorn rag_for_pandas.api:app

Interactive documentation is served at /docs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from rag_for_pandas.paths import FRONTEND_BUILD
from rag_for_pandas.pipeline import Pipeline, load_pipeline

MAX_QUERY_CHARS = 500
MAX_RESULTS = 20

# Surrounding whitespace is removed before the length check, so "   " is rejected as empty.
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)]


class SearchRequest(BaseModel):
    query: QueryText
    k: int = Field(default=5, ge=1, le=MAX_RESULTS, description="number of results")


class SearchResult(BaseModel):
    rank: int
    name: str
    score: float
    excerpt: str
    source_file: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AnswerRequest(BaseModel):
    question: QueryText


class AnswerSource(BaseModel):
    number: int
    name: str
    cited: bool


class AnswerResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    sources: list[AnswerSource]
    invalid_citations: list[int]


class HealthResponse(BaseModel):
    status: str
    documents: int
    answers_enabled: bool


def create_app(load: Callable[[], Pipeline] = load_pipeline, frontend: Path | None = FRONTEND_BUILD) -> FastAPI:
    """Build the app.

    Tests pass a `load` function that returns a pipeline with fake models. If
    `frontend` is a folder that exists, the built web page is served from it.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load models once at startup rather than on every request.
        app.state.pipeline = load()
        yield

    app = FastAPI(
        title="rag-for-pandas",
        description="Search pandas API documentation and get answers that cite it.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def pipeline(request: Request) -> Pipeline:
        return request.app.state.pipeline

    # Plain `def` endpoints run in a worker thread, so a slow model call does not
    # stop the server from accepting other requests.
    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        p = pipeline(request)
        return HealthResponse(status="ok", documents=len(p.docs), answers_enabled=p.can_answer)

    @app.post("/search")
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        hits = pipeline(request).search(body.query, body.k)
        return SearchResponse(query=body.query, results=[SearchResult(**asdict(hit)) for hit in hits])

    @app.post("/answer")
    def answer(body: AnswerRequest, request: Request) -> AnswerResponse:
        p = pipeline(request)
        if not p.can_answer:
            raise HTTPException(status_code=503, detail="answer generation is disabled on this server")
        result = p.answer(body.question)
        return AnswerResponse(
            question=body.question,
            answer=result.answer,
            abstained=result.abstained,
            sources=[AnswerSource(**asdict(source)) for source in result.sources],
            invalid_citations=result.invalid_citations,
        )

    # Mounted last: routes registered earlier (the API and /docs) are matched first,
    # so the page cannot hide them. html=True serves index.html for "/".
    if frontend is not None and frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()
