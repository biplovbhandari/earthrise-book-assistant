from __future__ import annotations

from typing import Protocol, runtime_checkable

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.document import Document


@runtime_checkable
class Parser(Protocol):
    """Reads a source file and produces a structured Document."""

    def parse(self, actual_path: str, source_path: str) -> Document:
        """Parse a source file into a Document.

        Args:
            actual_path: Filesystem path to the file (for reading).
            source_path: Repo-relative path (stored in metadata).

        Returns:
            Parsed document with title, content, and metadata.
        """
        ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Splits a Document into indexable Chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: Parsed document to chunk.

        Returns:
            List of chunks with content, metadata, and parent/child links.
        """
        ...
