from pydantic import BaseModel

from earthrise_rag.models.chunk import Chunk


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    ranking_method: str = "unknown"
