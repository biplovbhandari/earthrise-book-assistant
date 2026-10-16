from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import acquire_llm_slot, require_generation_ready, require_retrieval_ready
from api.routes.validators import QuestionFiltersMixin
from earthrise_rag.models.answer import Answer

logger = logging.getLogger(__name__)

router = APIRouter()


class AskRequest(QuestionFiltersMixin):
    """Incoming question with optional metadata filters."""


@router.post(
    "/ask",
    response_model=Answer,
    dependencies=[Depends(require_retrieval_ready), Depends(require_generation_ready)],
)
def ask(request: Request, body: AskRequest):
    """Generate an answer from the book using retrieval-augmented generation.

    Returns 503 if pipelines are not ready or if generation fails.
    """
    pipelines = request.app.state.pipelines

    llm_semaphore = acquire_llm_slot(request)
    try:
        result = pipelines.query.ask(body.question, body.filters)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=503, detail="generation failed")
    finally:
        if llm_semaphore is not None:
            llm_semaphore.release()

    return result
