from pydantic import BaseModel

from earthrise_rag.models.citation import Citation
from earthrise_rag.models.scored_chunk import ScoredChunk


class Answer(BaseModel):
    """LLM-generated answer with source chunks and citations."""

    answer: str
    sources: list[ScoredChunk]
    citations: list[Citation]
