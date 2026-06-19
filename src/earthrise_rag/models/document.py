from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    title: str
    source_path: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_type: Literal["book_text", "video_transcript"]
