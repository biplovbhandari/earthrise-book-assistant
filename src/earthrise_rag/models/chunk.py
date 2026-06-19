from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    content_hash: str
    source_type: Literal["book_text", "video_transcript"]
    content_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_type: Literal["parent", "child", "standalone"] = "standalone"
    parent_id: str | None = None
