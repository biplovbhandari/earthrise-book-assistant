from __future__ import annotations

from earthrise_rag.models.citation import Citation
from earthrise_rag.models.scored_chunk import ScoredChunk


class DefaultCitationBuilder:
    """Builds one citation per retrieved chunk from chunk metadata."""

    def build(self, chunks: list[ScoredChunk]) -> list[Citation]:
        """Build one citation per retrieved chunk from chunk metadata.

        Args:
            chunks: Ranked chunks from retrieval.

        Returns:
            List of citations in the same order as the input chunks.
        """
        citations = []
        for scored in chunks:
            meta = scored.chunk.metadata
            citations.append(
                Citation(
                    chunk_id=scored.chunk.id,
                    source_path=meta.get("source_path", ""),
                    chapter=meta.get("chapter", ""),
                    section=meta.get("section", ""),
                )
            )
        return citations
