from pydantic import BaseModel

from geoai_rag.models.citation import Citation
from geoai_rag.models.scored_chunk import ScoredChunk


class Answer(BaseModel):
    """LLM-generated answer with source chunks, citations, and stage timings.

    ``retrieval_ms`` and ``generation_ms`` are the elapsed milliseconds spent
    in retrieval and in the LLM call.  They default to 0 so callers that build
    an ``Answer`` without timings keep working.  A value of 0 also means the
    stage did not run, for example ``generation_ms`` when no chunks were found.
    """

    answer: str
    sources: list[ScoredChunk]
    citations: list[Citation]
    retrieval_ms: int = 0
    generation_ms: int = 0
