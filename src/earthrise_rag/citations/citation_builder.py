from __future__ import annotations

from urllib.parse import quote

from earthrise_rag.models.citation import Citation
from earthrise_rag.models.scored_chunk import ScoredChunk

_RENDERABLE_EXTENSIONS = (".ipynb", ".qmd", ".md")


class DefaultCitationBuilder:
    """Builds one citation per retrieved chunk from chunk metadata."""

    @staticmethod
    def _source_path_to_url(source_path: str) -> str:
        """Convert a repo-relative source path to a book HTML URL.

        Accepts only paths rooted at ``book/`` that end with a renderable
        extension.  The ``book/`` prefix is stripped, the file extension is
        swapped for ``.html``, and the remainder is percent-encoded and
        prepended with ``/``.

        Returns an empty string for any path that is absent, not under
        ``book/``, has a non-renderable extension, or contains traversal
        sequences, protocol-relative double-slashes, backslashes, or ASCII
        control characters.
        """
        if not source_path or not source_path.startswith("book/"):
            return ""

        path = source_path[5:]  # strip "book/" prefix

        if (
            path.startswith("/")
            or "//" in path
            or "\\" in path
            or ".." in path.split("/")
            or any(ord(c) < 32 or ord(c) == 127 for c in path)
        ):
            return ""

        if not any(path.endswith(ext) for ext in _RENDERABLE_EXTENSIONS):
            return ""

        for ext in _RENDERABLE_EXTENSIONS:
            if path.endswith(ext):
                path = path[: -len(ext)] + ".html"
                break

        path = quote(path, safe="/")
        return f"/{path}"

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
                    url=self._source_path_to_url(meta.get("source_path", "")),
                )
            )
        return citations
