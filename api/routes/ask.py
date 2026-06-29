from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from earthrise_rag.models.answer import Answer

logger = logging.getLogger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    """Incoming question with optional metadata filters."""

    question: str = Field(..., min_length=1, max_length=2000)
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


@router.post("/ask", response_model=Answer)
def ask(request: Request, body: AskRequest):
    """Generate an answer from the book using retrieval-augmented generation.

    Returns 503 if pipelines are not ready or if generation fails.
    """
    pipelines = getattr(request.app.state, "pipelines", None)
    if pipelines is None or pipelines.query is None:
        raise HTTPException(status_code=503, detail="generation not ready")

    try:
        if pipelines.vector_store is None or pipelines.vector_store.count() == 0:
            raise HTTPException(status_code=503, detail="retrieval not ready")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    try:
        result = pipelines.query.ask(body.question, body.filters)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=503, detail="generation failed")

    return result
