"""Public endpoint for thumbs up/down feedback on a recorded chat interaction.

The per-IP rate-limit increment and the feedback write share one transaction, so a submission
that fails does not use up quota.
Each is a single atomic statement rather than a read followed by a write, because with a read
first two concurrent requests could both pass the same check.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_db_session
from earthrise_rag.db.models import Feedback, FeedbackRateLimit, Interaction

router = APIRouter(tags=["feedback"])

MAX_COMMENT_LENGTH = 2000
RATE_LIMIT_WINDOW_MINUTES = 5
RATE_LIMIT_MAX_SUBMISSIONS = 10


class FeedbackRequest(BaseModel):
    """A reader's rating of one answer, with an optional comment explaining it."""

    interaction_id: UUID
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_LENGTH)


class FeedbackResponse(BaseModel):
    """The feedback now stored for the interaction."""

    interaction_id: str
    rating: str
    comment: str | None


def _hash_ip(ip: str) -> str:
    """Return the SHA-256 hex digest of a client IP so raw addresses are never stored."""
    return hashlib.sha256(ip.encode()).hexdigest()


def _window_start(now: datetime) -> datetime:
    """Round `now` down to the start of its rate-limit window.

    Counting from the Unix epoch gives every window the same length, even if the window length
    is changed to one that does not divide an hour.
    """
    window_seconds = RATE_LIMIT_WINDOW_MINUTES * 60
    epoch_seconds = int(now.timestamp())
    return datetime.fromtimestamp(epoch_seconds - epoch_seconds % window_seconds, tz=UTC)


async def _consume_rate_limit(session: AsyncSession, ip_hash: str, window_start: datetime) -> bool:
    """Count one submission against the IP's window; return False when the window is full.

    A single INSERT ... ON CONFLICT DO UPDATE ... WHERE makes the check and the increment one
    atomic step.
    When the window is full the WHERE is false, so nothing is updated and no row is returned.
    """
    stmt = (
        pg_insert(FeedbackRateLimit)
        .values(ip_hash=ip_hash, window_start=window_start, count=1)
        .on_conflict_do_update(
            index_elements=["ip_hash", "window_start"],
            set_={"count": FeedbackRateLimit.count + 1},
            where=FeedbackRateLimit.count < RATE_LIMIT_MAX_SUBMISSIONS,
        )
        .returning(FeedbackRateLimit.count)
    )
    return await session.scalar(stmt) is not None


async def _save_feedback(session: AsyncSession, body: FeedbackRequest) -> bool:
    """Store the rating; return True if it is the interaction's first, False if it replaced one.

    The INSERT goes first and the unique constraint on interaction_id decides whether it is a
    duplicate, so two concurrent first submissions cannot both be reported as created.
    A replacement overwrites rating and comment, so leaving the comment out clears it.
    admin_tags is never written here because only admins set it.
    """
    inserted_id = await session.scalar(
        pg_insert(Feedback)
        .values(interaction_id=body.interaction_id, rating=body.rating, comment=body.comment)
        .on_conflict_do_nothing(index_elements=["interaction_id"])
        .returning(Feedback.id)
    )
    if inserted_id is not None:
        return True
    await session.execute(
        update(Feedback)
        .where(Feedback.interaction_id == body.interaction_id)
        .values(rating=body.rating, comment=body.comment)
    )
    return False


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        201: {"model": FeedbackResponse, "description": "First rating of this interaction"},
        404: {"description": "No such interaction"},
        429: {"description": "Too many submissions from this IP address"},
    },
)
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(require_db_session)],
):
    """Record a reader's rating of an answer, replacing any earlier rating of the same answer.

    Responds 201 for the first rating of an interaction and 200 when it replaces an earlier one,
    so the widget can resend a changed rating without checking which case applies.
    """
    interaction_id = await session.scalar(
        select(Interaction.id).where(Interaction.id == body.interaction_id)
    )
    if interaction_id is None:
        raise HTTPException(status_code=404, detail="Interaction not found")

    client_ip = request.client.host if request.client else "unknown"
    ip_hash = _hash_ip(client_ip)
    window_start = _window_start(datetime.now(UTC))
    if not await _consume_rate_limit(session, ip_hash, window_start):
        raise HTTPException(
            status_code=429, detail="Too many feedback submissions. Please try again later."
        )

    created = await _save_feedback(session, body)
    await session.commit()

    response.status_code = 201 if created else 200
    return FeedbackResponse(
        interaction_id=str(body.interaction_id), rating=body.rating, comment=body.comment
    )
