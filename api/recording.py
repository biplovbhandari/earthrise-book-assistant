from __future__ import annotations

import logging
import posixpath
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from earthrise_rag.db.models.infrastructure import ChunkRecord
from earthrise_rag.db.models.interaction import (
    Conversation,
    Interaction,
    InteractionCitation,
    InteractionTrace,
)

logger = logging.getLogger(__name__)


@dataclass
class RecordingContext:
    """Accumulated data from a single chat interaction for DB recording."""

    interaction_id: uuid.UUID
    conversation_id: uuid.UUID
    visitor_id: uuid.UUID
    is_new_conversation: bool
    ga4_client_id: str | None
    question: str
    model_name: str
    response_text: str = ""
    token_event_count: int = 0
    latency_ms: int = 0
    retrieval_ms: int = 0
    generation_wall_ms: int = 0
    scored_chunks: list[dict] = field(default_factory=list)
    retrieval_query: str = ""
    saw_done: bool = False


def _filename_stem(source_path: str) -> str:
    """Return the filename component of *source_path* without its extension.

    Used as the last-resort display label when a chunk has neither a
    chapter nor a section in its metadata.
    """
    filename = source_path.rsplit("/", 1)[-1]
    stem, _, _ext = filename.rpartition(".")
    return stem or filename


def _source_path_to_book_url(source_path: str) -> str | None:
    """Convert a book-relative source path to an absolute book HTML URL.

    Only paths rooted at ``book/`` map to a rendered page. Everything else
    (video transcripts, unresolved sources) returns ``None`` so the caller
    can fall back to another URL source, such as a video watch link.
    """
    if not source_path.startswith("book/"):
        return None
    relative = source_path[len("book/") :]
    stem, _ext = posixpath.splitext(relative)
    return f"/{stem}.html"


def _derive_chunk_display(source_type: str, metadata: dict[str, Any]) -> tuple[str, str | None]:
    """Derive a chunk's display label and URL from its retrieval metadata.

    A simplified, cacheable version of the rules in
    ``DefaultCitationBuilder``: chapter and section combine into a label
    with no humanization, and a video transcript's watch link wins over a
    derived book URL when present.

    Args:
        source_type: The chunk's source type (e.g. "book_text", "video_transcript").
        metadata: The chunk's retrieval metadata dict.

    Returns:
        A (display_label, url) tuple. url is None when it cannot be derived.
    """
    source_path = metadata.get("source_path") or "unknown"
    chapter = metadata.get("chapter")
    section = metadata.get("section")

    if chapter and section:
        display_label = f"{chapter} - {section}"
    elif chapter:
        display_label = chapter
    elif section:
        display_label = section
    else:
        display_label = _filename_stem(source_path)

    watch_link = metadata.get("watch_link")
    if source_type == "video_transcript" and watch_link:
        url = watch_link
    else:
        url = _source_path_to_book_url(source_path)

    return display_label, url


def _build_chunk_and_citation_rows(
    scored_chunks: list[dict], interaction_id: uuid.UUID, index_run_id: int
) -> tuple[list[dict], list[dict]]:
    """Shape scored-chunk dicts into ChunkRecord and InteractionCitation row values.

    Pure and database-free so the retrieval-to-row mapping can be unit
    tested without a session. citation_index is each chunk's 0-based
    position in *scored_chunks*.
    """
    chunk_rows = []
    citation_rows = []
    for citation_index, chunk in enumerate(scored_chunks):
        metadata = chunk.get("metadata") or {}
        display_label, url = _derive_chunk_display(chunk.get("source_type", ""), metadata)
        chunk_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "index_run_id": index_run_id,
                "content": chunk["content"],
                "source_path": metadata.get("source_path") or "unknown",
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "url": url,
                "display_label": display_label,
            }
        )
        citation_rows.append(
            {
                "interaction_id": interaction_id,
                "citation_index": citation_index,
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "ranking_method": chunk["ranking_method"],
            }
        )
    return chunk_rows, citation_rows


def _build_trace_rows(ctx: RecordingContext) -> list[dict]:
    """Build the retrieval-stage and generation-stage InteractionTrace row values."""
    return [
        {
            "interaction_id": ctx.interaction_id,
            "stage": "retrieval",
            "latency_ms": ctx.retrieval_ms,
            "data": {
                "chunk_count": len(ctx.scored_chunks),
                "retrieval_query": ctx.retrieval_query,
            },
        },
        {
            "interaction_id": ctx.interaction_id,
            "stage": "generation",
            "latency_ms": ctx.generation_wall_ms,
            "data": {
                "token_event_count": ctx.token_event_count,
                "model": ctx.model_name,
            },
        },
    ]


async def record_interaction(
    session_factory: async_sessionmaker[AsyncSession],
    ctx: RecordingContext,
    *,
    embedding: list[float],
    deployment_id: int,
    index_run_id: int,
) -> None:
    """Persist one chat interaction and its retrieval trail in a single transaction.

    Opens its own session from *session_factory* rather than relying on a
    request-scoped dependency, because this is invoked from a FastAPI
    BackgroundTask that may run after the originating request's session has
    already been torn down. Within one transaction: upserts the conversation,
    backfills a ChunkRecord for any newly-cited chunk (so the citation
    foreign key always resolves), inserts the interaction, its per-chunk
    citations, and its retrieval/generation traces, then commits.

    Never raises. Any failure -- database unavailable, a constraint
    violation, malformed scored-chunk data -- is logged and swallowed so a
    recording bug can never affect the chat response already delivered to
    the user.

    Args:
        session_factory: Factory for a new AsyncSession, independent of any
            request-scoped session.
        ctx: Accumulated interaction data collected during the chat request.
        embedding: The dense query embedding for this interaction's question.
        deployment_id: Active Deployment row this interaction is attributed to.
        index_run_id: Active IndexRun row any newly-recorded chunks belong to.
    """
    try:
        async with session_factory() as session:
            await session.execute(
                insert(Conversation)
                .values(
                    id=ctx.conversation_id,
                    visitor_id=ctx.visitor_id,
                    ga4_client_id=ctx.ga4_client_id,
                )
                .on_conflict_do_nothing()
            )

            chunk_rows, citation_rows = _build_chunk_and_citation_rows(
                ctx.scored_chunks, ctx.interaction_id, index_run_id
            )

            if chunk_rows:
                await session.execute(
                    insert(ChunkRecord).values(chunk_rows).on_conflict_do_nothing()
                )

            await session.execute(
                insert(Interaction).values(
                    id=ctx.interaction_id,
                    conversation_id=ctx.conversation_id,
                    question=ctx.question,
                    response_text=ctx.response_text,
                    token_count=ctx.token_event_count,
                    latency_ms=ctx.latency_ms,
                    query_embedding=embedding,
                    deployment_id=deployment_id,
                )
            )

            if citation_rows:
                await session.execute(insert(InteractionCitation).values(citation_rows))

            await session.execute(insert(InteractionTrace).values(_build_trace_rows(ctx)))

            await session.commit()
    except Exception:
        logger.exception("Failed to record chat interaction %s", ctx.interaction_id)
