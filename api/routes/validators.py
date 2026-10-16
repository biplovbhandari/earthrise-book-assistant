from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QuestionFiltersMixin(BaseModel):
    """Base model with shared question and filters fields."""

    question: str = Field(..., min_length=1, max_length=2000)
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
