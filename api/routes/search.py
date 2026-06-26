from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from earthrise_rag.models.scored_chunk import ScoredChunk

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Incoming search query with optional top_k and metadata filters."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    filters: dict[str, str | int | bool] | None = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, v):
        if isinstance(v, str):
            v = v.strip()
        return v

    @field_validator("filters", mode="before")
    @classmethod
    def reject_dotted_keys(cls, v):
        if v is not None:
            for key in v:
                if "." in key:
                    raise ValueError(
                        f"Filter key '{key}' must not contain '.';"
                        " use bare key names (e.g. 'chapter' not 'metadata.chapter')"
                    )
        return v


class SearchResponse(BaseModel):
    """Search results containing ranked chunks."""

    chunks: list[ScoredChunk]


@router.post("/search", response_model=SearchResponse)
def search(request: Request, body: SearchRequest):
    pipelines = getattr(request.app.state, "pipelines", None)

    if pipelines is None or pipelines.query is None:
        raise HTTPException(status_code=503, detail="retrieval not ready")

    try:
        if pipelines.vector_store is None or pipelines.vector_store.count() == 0:
            raise HTTPException(status_code=503, detail="vector store not ready")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    top_k = body.top_k if body.top_k is not None else request.app.state.settings.retrieval_top_k

    try:
        results = pipelines.query.search(body.question, top_k, body.filters)
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    return SearchResponse(chunks=results)
