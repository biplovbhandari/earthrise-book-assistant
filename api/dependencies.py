from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy.ext.asyncio import AsyncSession

from earthrise_rag.factories import Pipelines  # noqa: F401 -- re-exported for test imports

logger = logging.getLogger(__name__)

_QDRANT_ERRORS = (UnexpectedResponse,)


def require_retrieval_ready(request: Request) -> None:
    """Verify vector store is populated. Raises 503 if not."""
    pipelines = getattr(request.app.state, "pipelines", None)
    if pipelines is None or getattr(pipelines, "vector_store", None) is None:
        raise HTTPException(status_code=503, detail="retrieval not ready")
    try:
        if pipelines.vector_store.count() == 0:
            raise HTTPException(status_code=503, detail="retrieval not ready")
    except (ConnectionError, OSError, RuntimeError):
        raise HTTPException(status_code=503, detail="retrieval not ready")
    except _QDRANT_ERRORS:
        raise HTTPException(status_code=503, detail="retrieval not ready")


def require_generation_ready(request: Request) -> None:
    """Verify all generation adapters are present. Raises 503 if not."""
    pipelines = getattr(request.app.state, "pipelines", None)
    if pipelines is None or getattr(pipelines, "query", None) is None:
        raise HTTPException(status_code=503, detail="pipelines not initialized")
    if not pipelines.query.generation_ready:
        raise HTTPException(status_code=503, detail="generation adapters incomplete")


def check_retrieval_ready(pipelines) -> tuple[bool, str]:
    """Check vector store is populated. Returns (ready, reason)."""
    if pipelines is None or getattr(pipelines, "vector_store", None) is None:
        return False, "retrieval not ready"
    try:
        if pipelines.vector_store.count() == 0:
            return False, "retrieval not ready"
    except (ConnectionError, OSError, RuntimeError):
        return False, "retrieval not ready"
    except _QDRANT_ERRORS:
        return False, "retrieval not ready"
    return True, ""


def check_generation_ready(pipelines) -> tuple[bool, str]:
    """Check all generation adapters are present. Returns (ready, reason)."""
    if pipelines is None or getattr(pipelines, "query", None) is None:
        return False, "pipelines not initialized"
    if not pipelines.query.generation_ready:
        return False, "generation adapters incomplete"
    return True, ""


def acquire_llm_slot(request: Request) -> threading.Semaphore | None:
    """Acquire the LLM concurrency semaphore or raise 503.

    Returns the acquired semaphore for the caller to release, or None
    if no semaphore is configured.
    """
    semaphore = getattr(request.app.state, "llm_semaphore", None)
    if semaphore is not None and not semaphore.acquire(timeout=120):
        raise HTTPException(status_code=503, detail="LLM busy. Please try again later.")
    return semaphore


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession | None, None]:
    """Yield a DB session or None. For routes that optionally record."""
    factory = getattr(request.app.state, "db_session_factory", None)
    if factory is None:
        yield None
        return
    async with factory() as session:
        yield session


async def require_admin_token(request: Request) -> None:
    """Verify the Authorization header carries a valid admin Bearer token.

    Raises 401 if the token is missing or wrong, and 403 if ADMIN_TOKEN
    is not configured (prevents silent open access on misconfiguration).
    """
    from earthrise_rag.config import get_settings

    expected = get_settings().admin_token.get_secret_value()
    if not expected:
        raise HTTPException(status_code=403, detail="Admin access not configured")
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    # compare_digest keeps the check constant-time. It needs bytes because it raises TypeError
    # on a str with non-ASCII characters, and a client can send those in a header, which would
    # turn a bad token into a 500 instead of a 401.
    if not secrets.compare_digest(auth[7:].encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid token")


async def require_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session or raise 503. For admin routes that require DB."""
    from sqlalchemy.exc import InterfaceError, OperationalError, TimeoutError

    factory = getattr(request.app.state, "db_session_factory", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        async with factory() as session:
            yield session
    except (OperationalError, InterfaceError, TimeoutError, OSError):
        raise HTTPException(status_code=503, detail="Database unavailable")


DbSession = Annotated[AsyncSession, Depends(require_db_session)]
