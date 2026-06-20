from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.document import Document
from earthrise_rag.models.scored_chunk import ScoredChunk


@runtime_checkable
class Parser(Protocol):
    def parse(self, actual_path: str, source_path: str) -> Document: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def get_dimension(self) -> int: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search_dense(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...

    def search_sparse(
        self,
        text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...

    def get_by_ids(self, ids: list[str]) -> list[Chunk]: ...

    def delete_by_source(self, source_path: str) -> None: ...
