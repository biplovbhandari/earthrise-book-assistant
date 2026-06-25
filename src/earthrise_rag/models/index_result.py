from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class IndexResult(BaseModel):
    """Result of indexing a single source file."""

    source_path: str
    status: Literal["success", "skipped", "failed"]
    chunks_indexed: int = 0
    error: str | None = None
    embeddings_model: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
