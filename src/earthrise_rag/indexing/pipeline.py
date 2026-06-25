from __future__ import annotations

import logging
from pathlib import Path

from earthrise_rag.indexing.base import ChunkingStrategy, Parser
from earthrise_rag.interfaces import Embedder, VectorStore
from earthrise_rag.models.index_result import IndexResult

logger = logging.getLogger(__name__)

_STUB_PHRASE = "under development"
_STUB_WORD_LIMIT = 50


class IndexingPipeline:
    """Orchestrates parse → chunk → embed → store for a single source file."""

    def __init__(
        self,
        parsers: dict[str, Parser],
        chunkers: dict[str, ChunkingStrategy],
        embedder: Embedder,
        vector_store: VectorStore,
        commit_sha: str = "",
    ) -> None:
        self.parsers = parsers
        self.chunkers = chunkers
        self.embedder = embedder
        self.vector_store = vector_store
        self.commit_sha = commit_sha

    def index_source(self, actual_path: str, source_path: str) -> IndexResult:
        """Parse, chunk, embed, and upsert a single source file.

        Args:
            actual_path: Filesystem path to read the file.
            source_path: Repo-relative path stored in metadata.

        Returns:
            IndexResult with status, chunk count, and any error.
        """
        ext = Path(actual_path).suffix.lower()

        if ext not in self.parsers:
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported: {list(self.parsers.keys())}"
            )

        parser = self.parsers[ext]
        chunker = self.chunkers[ext]

        document = parser.parse(actual_path, source_path)
        document.metadata["commit_sha"] = self.commit_sha

        if (
            _STUB_PHRASE in document.content.lower()
            and len(document.content.split()) < _STUB_WORD_LIMIT
        ):
            logger.info("Skipping stub: %s", source_path)
            return IndexResult(source_path=source_path, status="skipped")

        chunks = chunker.chunk(document)
        if not chunks:
            return IndexResult(source_path=source_path, status="skipped")

        texts = [c.content for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks in {source_path}"
            )

        self.vector_store.delete_by_source(source_path)
        self.vector_store.upsert(chunks, vectors)

        return IndexResult(
            source_path=source_path,
            status="success",
            chunks_indexed=len(chunks),
            embeddings_model=getattr(self.embedder, "_model_name", ""),
        )
