from __future__ import annotations

from typing import Any

from earthrise_rag.interfaces import RetrievalStrategy
from earthrise_rag.models.scored_chunk import ScoredChunk


class QueryPipeline:
    """Composition root for query operations (search, and later ask)."""

    def __init__(self, strategy: RetrievalStrategy) -> None:
        self._strategy = strategy

    def search(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search for relevant chunks using the configured retrieval strategy.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        return self._strategy.retrieve(question, top_k, filters)
