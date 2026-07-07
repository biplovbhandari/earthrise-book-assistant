from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CONTENT = 8000


class ChatMessage(BaseModel):
    """A single conversation turn in the chat history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Incoming chat question with optional history and filters."""

    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    filters: dict[str, str | int | bool] | None = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, v):
        """Strip leading and trailing whitespace from the question."""
        if isinstance(v, str):
            v = v.strip()
        return v

    @field_validator("filters", mode="before")
    @classmethod
    def reject_dotted_keys(cls, v):
        """Reject filter keys that contain dots to prevent metadata path confusion."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("filters must be an object")
            for key in v:
                if "." in key:
                    raise ValueError(
                        f"Filter key '{key}' must not contain '.';"
                        " use bare key names (e.g. 'chapter' not 'metadata.chapter')"
                    )
        return v

    @field_validator("history", mode="before")
    @classmethod
    def truncate_history(cls, v):
        """Truncate history: cap count, cap per-entry content, drop orphan leading assistant."""
        if not isinstance(v, list):
            return v
        if len(v) > MAX_HISTORY_MESSAGES:
            v = v[-MAX_HISTORY_MESSAGES:]
        for entry in v:
            if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                if len(entry["content"]) > MAX_HISTORY_CONTENT:
                    entry["content"] = entry["content"][:MAX_HISTORY_CONTENT]
        while v and isinstance(v[0], dict) and v[0].get("role") == "assistant":
            v = v[1:]
        return v


def check_retrieval_ready(pipelines) -> tuple[bool, str]:
    """Check vector store is populated.

    Returns a (ready, reason) tuple where reason is empty when ready is True.
    """
    if pipelines is None or getattr(pipelines, "vector_store", None) is None:
        return False, "retrieval not ready"
    try:
        if pipelines.vector_store.count() == 0:
            return False, "retrieval not ready"
    except Exception:
        return False, "retrieval not ready"
    return True, ""


def check_generation_ready(pipelines) -> tuple[bool, str]:
    """Check all generation adapters are present.

    Returns a (ready, reason) tuple where reason is empty when ready is True.
    """
    if pipelines is None or getattr(pipelines, "query", None) is None:
        return False, "pipelines not initialized"
    q = pipelines.query
    if q._context_builder is None or q._llm_client is None or q._citation_builder is None:
        return False, "generation adapters incomplete"
    return True, ""


def check_chat_ready(pipelines) -> tuple[bool, str]:
    """Check retrieval + generation + streaming readiness.

    Returns a (ready, reason) tuple where reason is empty when ready is True.
    """
    ready, reason = check_retrieval_ready(pipelines)
    if not ready:
        return ready, reason
    ready, reason = check_generation_ready(pipelines)
    if not ready:
        return ready, reason
    if not callable(getattr(pipelines.query._llm_client, "chat_stream", None)):
        return False, "streaming not supported"
    return True, ""


@router.post("/chat")
def chat(request: Request, body: ChatRequest):
    """Stream a RAG answer as server-sent events.

    Returns 503 if pipelines are not ready or streaming is unsupported.
    Each SSE event is a JSON-encoded dict emitted by the query pipeline's ask_stream method.
    """
    pipelines = getattr(request.app.state, "pipelines", None)
    ready, reason = check_chat_ready(pipelines)
    if not ready:
        raise HTTPException(status_code=503, detail=reason)
    assert pipelines is not None and pipelines.query is not None

    history_dicts = [{"role": m.role, "content": m.content} for m in body.history]

    def event_stream():
        """Yield SSE-formatted events from the streaming pipeline."""
        try:
            for event in pipelines.query.ask_stream(
                body.question, history=history_dicts, filters=body.filters
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming generation failed")
            error = {"type": "error", "message": "Generation failed. Please try again."}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
