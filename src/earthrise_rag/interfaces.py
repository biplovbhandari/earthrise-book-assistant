from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.scored_chunk import ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Converts text into dense vector embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text.

        Returns:
            Embedding vector.
        """
        ...

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent store for chunk vectors and payloads."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks with their embedding vectors.

        Args:
            chunks: Chunk objects (payload stored alongside vectors).
            vectors: Corresponding embedding vectors (same length as chunks).
        """
        ...

    def search_dense(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by dense vector similarity.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        ...

    def search_sparse(
        self,
        text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by sparse/keyword matching (e.g. BM25).

        Args:
            text: Raw query text.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        ...

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs.

        Returns:
            List of matching chunks.
        """
        ...

    def delete_by_source(self, source_path: str) -> None:
        """Delete all chunks originating from a given source file.

        Args:
            source_path: Repo-relative path (e.g. "book/03_Semantic_Segmentation/...").
        """
        ...

    def count(self) -> int:
        """Return the approximate number of points in the collection."""
        ...


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Retrieves relevant chunks for a query."""

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve ranked chunks for a natural language question.

        Args:
            question: The user's query.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores and re-orders retrieval candidates."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Rerank candidates and return the top results.

        Args:
            query: The original query text.
            candidates: Chunks from the initial retrieval.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of scored chunks.
        """
        ...
