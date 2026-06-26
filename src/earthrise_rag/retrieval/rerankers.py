from __future__ import annotations

from earthrise_rag.models.scored_chunk import ScoredChunk


class NoOpReranker:
    """Passthrough reranker that preserves the original ranking order."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        return candidates[:top_k]
