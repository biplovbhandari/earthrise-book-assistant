from __future__ import annotations

from typing import Any

from earthrise_rag.interfaces import Embedder, Reranker, VectorStore
from earthrise_rag.models.scored_chunk import ScoredChunk


class HybridStrategy:
    """Retrieval via dense + sparse search fused with Reciprocal Rank Fusion."""

    _OVERSAMPLING = 3

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        rrf_k: int = 60,
    ) -> None:
        """Wire the retrieval components and RRF tuning constant."""
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker
        self._rrf_k = rrf_k

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Fetch from dense and sparse indexes, fuse via RRF, then rerank.

        When sparse returns empty (old collection, no terms produced),
        skips RRF and returns dense results with original scores and
        ranking_method unchanged -- no score rewriting into RRF fractions.
        """
        fetch_k = min(top_k * self._OVERSAMPLING, 50)

        vector = self._embedder.embed_query(question)
        dense_results = self._store.search_dense(vector, fetch_k, filters)
        sparse_results = self._store.search_sparse(question, fetch_k, filters)

        if not sparse_results:
            return self._reranker.rerank(question, dense_results[:top_k], top_k)

        fused = self._rrf_fuse([dense_results, sparse_results], top_k)
        return self._reranker.rerank(question, fused, top_k)

    def _rrf_fuse(self, results_lists: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion: score = sum of 1/(k + rank) per list.

        Ranks are 1-indexed. Chunks appearing in multiple lists get their
        scores summed, which boosts items found by both dense and sparse.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, Any] = {}

        for results in results_lists:
            for rank, sc in enumerate(results, start=1):
                cid = sc.chunk.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                chunks[cid] = sc.chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [
            ScoredChunk(chunk=chunks[cid], score=scores[cid], ranking_method="hybrid_rrf")
            for cid in sorted_ids[:top_k]
        ]


class DenseStrategy:
    """Retrieval via dense vector similarity search."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker) -> None:
        """Wire embedder, store, and reranker for the dense retrieval path."""
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
