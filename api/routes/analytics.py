"""Read-only analytics endpoints backed by the Postgres reporting views.

The views are plain aggregates with no ORM models, so each endpoint queries one with text() and
maps the rows onto a Pydantic response model.
Each query states its own ORDER BY instead of relying on the view's, because a view's ordering
is not guaranteed once the view is wrapped in another query.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_admin_token, require_db_session

router = APIRouter(
    prefix="/analytics", tags=["analytics"], dependencies=[Depends(require_admin_token)]
)


class OverviewStats(BaseModel):
    """All-time totals across recorded conversations and interactions."""

    total_interactions: int
    total_conversations: int
    unique_visitors: int
    first_interaction_at: datetime | None
    last_interaction_at: datetime | None
    avg_latency_ms: float | None


_OVERVIEW_SQL = """
    SELECT
        (SELECT COUNT(*) FROM earthrise.interactions) AS total_interactions,
        (SELECT COUNT(*) FROM earthrise.conversations) AS total_conversations,
        (SELECT COUNT(DISTINCT visitor_id) FROM earthrise.conversations) AS unique_visitors,
        (SELECT MIN(created_at) FROM earthrise.interactions) AS first_interaction_at,
        (SELECT MAX(created_at) FROM earthrise.interactions) AS last_interaction_at,
        (SELECT AVG(latency_ms) FROM earthrise.interactions) AS avg_latency_ms
"""


@router.get("/overview", response_model=OverviewStats)
async def overview(session: Annotated[AsyncSession, Depends(require_db_session)]):
    """Return all-time totals, the interaction date range, and mean latency."""
    result = await session.execute(text(_OVERVIEW_SQL))
    return OverviewStats.model_validate(dict(result.mappings().one()))


class DailyStat(BaseModel):
    """Usage for one calendar day."""

    day: date
    total_queries: int
    thumbs_up: int
    thumbs_down: int
    avg_tokens: float | None
    avg_latency_ms: float | None


# The CAST pins the parameter to integer so Postgres resolves "date - integer" instead of
# guessing the type of an untyped parameter.
_DAILY_SQL = """
    SELECT day, total_queries, thumbs_up, thumbs_down, avg_tokens, avg_latency_ms
    FROM earthrise.v_daily_stats
    WHERE day > CURRENT_DATE - CAST(:days AS integer)
    ORDER BY day DESC
"""


@router.get("/daily", response_model=list[DailyStat])
async def daily(
    session: Annotated[AsyncSession, Depends(require_db_session)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
):
    """Return per-day usage for the last `days` calendar days, today included, newest first."""
    result = await session.execute(text(_DAILY_SQL), {"days": days})
    return [DailyStat.model_validate(dict(row)) for row in result.mappings().all()]


class CitationStat(BaseModel):
    """A cited source section and how readers rated the answers that cited it.

    thumbs_up_pct is a fraction from 0.0 to 1.0, not a percentage.
    It is None when none of the answers citing the source has been rated.
    """

    source_path: str
    display_label: str | None
    cite_count: int
    thumbs_up_pct: float | None


_CITATIONS_SQL = """
    SELECT source_path, display_label, cite_count, thumbs_up_pct
    FROM earthrise.v_citation_heatmap
    ORDER BY cite_count DESC, source_path, display_label
    LIMIT :limit
"""


@router.get("/citations", response_model=list[CitationStat])
async def citations(
    session: Annotated[AsyncSession, Depends(require_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """Return the most cited source sections, most cited first."""
    result = await session.execute(text(_CITATIONS_SQL), {"limit": limit})
    return [CitationStat.model_validate(dict(row)) for row in result.mappings().all()]


class RetrievalGap(BaseModel):
    """A question that received thumbs-down feedback, and how well retrieval scored for it.

    avg_top_score averages the best citation score of each answer.
    It is None when none of the answers cited any chunk.
    """

    question: str
    avg_top_score: float | None
    thumbs_down_count: int


_GAPS_SQL = """
    SELECT question, avg_top_score, thumbs_down_count
    FROM earthrise.v_retrieval_gaps
    ORDER BY avg_top_score ASC NULLS FIRST, question
    LIMIT :limit
"""


@router.get("/gaps", response_model=list[RetrievalGap])
async def gaps(
    session: Annotated[AsyncSession, Depends(require_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """Return thumbs-down questions, lowest retrieval score first."""
    result = await session.execute(text(_GAPS_SQL), {"limit": limit})
    return [RetrievalGap.model_validate(dict(row)) for row in result.mappings().all()]


class ConversationSummary(BaseModel):
    """One conversation with its turn count and how long it ran.

    duration_seconds runs from the conversation's creation to its last interaction.
    It is None when the conversation has no interactions.
    """

    id: str
    title: str | None
    topic: str | None
    visitor_id: str | None
    created_at: datetime
    interaction_count: int
    last_activity: datetime | None
    duration_seconds: float | None


_CONVERSATIONS_SQL = """
    SELECT id, title, topic, visitor_id, created_at, interaction_count, last_activity, duration
    FROM earthrise.v_conversation_summary
    ORDER BY created_at DESC, id DESC
    LIMIT :limit
"""


def _conversation_from_row(row: RowMapping) -> ConversationSummary:
    """Convert a view row: UUIDs become strings and the interval becomes seconds."""
    visitor_id = row["visitor_id"]
    duration = row["duration"]
    return ConversationSummary(
        id=str(row["id"]),
        title=row["title"],
        topic=row["topic"],
        visitor_id=None if visitor_id is None else str(visitor_id),
        created_at=row["created_at"],
        interaction_count=row["interaction_count"],
        last_activity=row["last_activity"],
        duration_seconds=None if duration is None else duration.total_seconds(),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def conversations(
    session: Annotated[AsyncSession, Depends(require_db_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    """Return the most recently started conversations, newest first."""
    result = await session.execute(text(_CONVERSATIONS_SQL), {"limit": limit})
    return [_conversation_from_row(row) for row in result.mappings().all()]
