import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from earthrise_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from earthrise_rag.db.models.interaction import Interaction


class SharedResponse(TimestampMixin, Base):
    """Shareable link for a specific Q&A interaction."""

    __tablename__ = "shared_responses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    interaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"), unique=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    interaction: Mapped["Interaction"] = relationship(back_populates="shared_response")


class FeedbackRateLimit(Base):
    """Track feedback submission counts per IP per time window for rate limiting."""

    __tablename__ = "feedback_rate_limits"
    __table_args__ = (CheckConstraint("count >= 0", name="count_nonneg"),)

    ip_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    count: Mapped[int]
