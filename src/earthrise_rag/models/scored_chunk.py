from pydantic import BaseModel

from earthrise_rag.models.chunk import Chunk


class ScoredChunk(BaseModel):
    """A chunk with a relevance score from retrieval or reranking."""

    chunk: Chunk
    score: float
    ranking_method: str = "unknown"
