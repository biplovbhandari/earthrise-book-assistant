from __future__ import annotations

from pydantic import field_validator


class QuestionFiltersMixin:
    """Shared validators for question and filters fields."""

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
