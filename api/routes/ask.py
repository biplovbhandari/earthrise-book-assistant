from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.routes.chat import check_generation_ready, check_retrieval_ready
from api.routes.validators import QuestionFiltersMixin
from earthrise_rag.models.answer import Answer

logger = logging.getLogger(__name__)

router = APIRouter()


class AskRequest(QuestionFiltersMixin, BaseModel):
    """Incoming question with optional metadata filters."""

    question: str = Field(..., min_length=1, max_length=2000)
    filters: dict[str, str | int | bool] | None = None


@router.post("/ask", response_model=Answer)
def ask(request: Request, body: AskRequest):
    """Generate an answer from the book using retrieval-augmented generation.

    Returns 503 if pipelines are not ready or if generation fails.
    """
    pipelines = getattr(request.app.state, "pipelines", None)
    ready, reason = check_retrieval_ready(pipelines)
    if not ready:
        raise HTTPException(status_code=503, detail=reason)
    ready, reason = check_generation_ready(pipelines)
    if not ready:
        raise HTTPException(status_code=503, detail=reason)
    assert pipelines is not None and pipelines.query is not None

    try:
        result = pipelines.query.ask(body.question, body.filters)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=503, detail="generation failed")

    return result
