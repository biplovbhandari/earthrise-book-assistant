from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.dependencies import require_retrieval_ready
from api.metrics import retrieval_chunk_count, search_request_count
from api.routes.validators import QuestionFiltersMixin
from earthrise_rag.config import TOP_K_MAX, TOP_K_MIN
from earthrise_rag.models.scored_chunk import ScoredChunk

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(QuestionFiltersMixin):
    """Incoming search query with optional top_k and metadata filters."""

    top_k: int | None = Field(None, ge=TOP_K_MIN, le=TOP_K_MAX)


class SearchResponse(BaseModel):
    """Search results containing ranked chunks."""

    chunks: list[ScoredChunk]


@router.post(
    "/search", response_model=SearchResponse, dependencies=[Depends(require_retrieval_ready)]
)
def search(request: Request, body: SearchRequest):
    pipelines = request.app.state.pipelines

    top_k = body.top_k if body.top_k is not None else request.app.state.settings.retrieval_top_k

    try:
        results = pipelines.query.search(body.question, top_k, body.filters)
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    search_request_count.inc()
    retrieval_chunk_count.observe(len(results))
    return SearchResponse(chunks=results)
