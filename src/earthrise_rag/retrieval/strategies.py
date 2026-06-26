from __future__ import annotations

from typing import Any

from earthrise_rag.interfaces import Embedder, Reranker, VectorStore
from earthrise_rag.models.scored_chunk import ScoredChunk


class DenseStrategy:
    """Retrieval via dense vector similarity search."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Embed query, search dense vectors, and rerank results.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g. {"chapter": "03"}).

        Returns:
            Ranked list of scored chunks.
        """
        vector = self._embedder.embed_query(question)
        candidates = self._store.search_dense(vector, top_k, filters)
        return self._reranker.rerank(question, candidates, top_k)
