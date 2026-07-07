from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.routes.chat import check_retrieval_ready
from api.routes.validators import QuestionFiltersMixin
from earthrise_rag.models.scored_chunk import ScoredChunk

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(QuestionFiltersMixin, BaseModel):
    """Incoming search query with optional top_k and metadata filters."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    filters: dict[str, str | int | bool] | None = None


class SearchResponse(BaseModel):
    """Search results containing ranked chunks."""

    chunks: list[ScoredChunk]


@router.post("/search", response_model=SearchResponse)
def search(request: Request, body: SearchRequest):
    pipelines = getattr(request.app.state, "pipelines", None)
    ready, reason = check_retrieval_ready(pipelines)
    if not ready:
        raise HTTPException(status_code=503, detail=reason)
    if pipelines is None or pipelines.query is None:
        raise HTTPException(status_code=503, detail="retrieval not ready")

    top_k = body.top_k if body.top_k is not None else request.app.state.settings.retrieval_top_k

    try:
        results = pipelines.query.search(body.question, top_k, body.filters)
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    return SearchResponse(chunks=results)
