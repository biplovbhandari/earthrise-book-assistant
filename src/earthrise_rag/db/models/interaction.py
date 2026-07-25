import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from earthrise_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from earthrise_rag.db.models.infrastructure import ChunkRecord, Deployment
    from earthrise_rag.db.models.sharing import SharedResponse


class Conversation(TimestampMixin, Base):
    """A chat thread grouping related Q&A interactions from one user intent."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    visitor_id: Mapped[uuid.UUID | None] = mapped_column(index=True)
    ga4_client_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="conversation", cascade="all, delete", passive_deletes=True
    )


class Interaction(TimestampMixin, Base):
    """One Q&A exchange - a question from the user and the LLM's response."""

    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint("token_count >= 0", name="token_count_nonneg"),
        CheckConstraint("latency_ms >= 0", name="latency_nonneg"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]
    latency_ms: Mapped[int]
    query_embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    deployment_id: Mapped[int] = mapped_column(
        ForeignKey("deployments.id", ondelete="RESTRICT"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="interactions")
    deployment: Mapped["Deployment"] = relationship(back_populates="interactions")
    citations: Mapped[list["InteractionCitation"]] = relationship(
        back_populates="interaction", cascade="all, delete", passive_deletes=True
    )
    traces: Mapped[list["InteractionTrace"]] = relationship(
        back_populates="interaction", cascade="all, delete", passive_deletes=True
    )
    feedback: Mapped["Feedback | None"] = relationship(
        back_populates="interaction", cascade="all, delete", passive_deletes=True
    )
    shared_response: Mapped["SharedResponse | None"] = relationship(
        back_populates="interaction", cascade="all, delete", passive_deletes=True
    )


class InteractionCitation(Base):
    """Records which chunks were cited in a response."""

    __tablename__ = "interaction_citations"
    __table_args__ = (
        UniqueConstraint("interaction_id", "citation_index"),
        CheckConstraint("citation_index >= 0", name="citation_index_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    interaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"), index=True
    )
    citation_index: Mapped[int]
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="RESTRICT"), index=True
    )
    score: Mapped[float]
    ranking_method: Mapped[str] = mapped_column(Text)

    interaction: Mapped["Interaction"] = relationship(back_populates="citations")
    chunk: Mapped["ChunkRecord"] = relationship(back_populates="citations")


class InteractionTrace(Base):
    """Raw pipeline state at each stage."""

    __tablename__ = "interaction_traces"
    __table_args__ = (
        UniqueConstraint("interaction_id", "stage"),
        CheckConstraint("latency_ms >= 0", name="latency_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    interaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(Text, index=True)
    data: Mapped[Any] = mapped_column(JSONB)
    latency_ms: Mapped[int]

    interaction: Mapped["Interaction"] = relationship(back_populates="traces")


class Feedback(TimestampMixin, Base):
    """User's rating of a response, plus admin classification."""

    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("interaction_id"),
        CheckConstraint("rating IN ('up', 'down')", name="rating_valid"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    interaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"), index=True
    )
    rating: Mapped[str] = mapped_column(Text, index=True)
    comment: Mapped[str | None] = mapped_column(Text)
    admin_tags: Mapped[Any] = mapped_column(JSONB, nullable=True)

    interaction: Mapped["Interaction"] = relationship(back_populates="feedback")
