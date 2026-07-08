from __future__ import annotations

import re
from urllib.parse import quote

from earthrise_rag.models.citation import Citation
from earthrise_rag.models.scored_chunk import ScoredChunk

_RENDERABLE_EXTENSIONS = (".ipynb", ".qmd", ".md")
_YOUTUBE_WATCH_RE = re.compile(r"https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]+(&t=\d+s)?")
_LEADING_DIGITS_RE = re.compile(r"^\d+_+")


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

    @staticmethod
    def _humanize_dir_name(name: str) -> str:
        """Strip leading digit-underscore prefix and replace underscores with spaces."""
        if not name:
            return ""
        cleaned = _LEADING_DIGITS_RE.sub("", name)
        return " ".join(cleaned.replace("_", " ").split())

    @staticmethod
    def _filename_label(source_path: str) -> str:
        """Titlecased filename stem from a source path."""
        if not source_path:
            return ""
        filename = source_path.rsplit("/", 1)[-1]
        stem, _, _ = filename.rpartition(".")
        if not stem:
            return ""
        return stem.replace("_", " ").title()

    @staticmethod
    def _build_display_label(
        chapter: str,
        section: str,
        source_path: str,
        source_type: str,
    ) -> str:
        """Build a human-readable citation label using provenance-aware rules."""
        ch = DefaultCitationBuilder._humanize_dir_name(chapter)
        sec = (
            DefaultCitationBuilder._humanize_dir_name(section)
            if source_type == "video_transcript"
            else section
        )
        if ch and sec:
            return f"{ch} - {sec}"
        if ch:
            filename = DefaultCitationBuilder._filename_label(source_path)
            return f"{ch} - {filename}" if filename else ch
        if sec:
            return sec
        return DefaultCitationBuilder._filename_label(source_path) or "Source"

    @staticmethod
    def _watch_link_url(watch_link: str) -> str:
        """Return *watch_link* if it is a valid timestamped YouTube URL, else ``""``."""
        if watch_link and _YOUTUBE_WATCH_RE.fullmatch(watch_link):
            return watch_link
        return ""

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
            source_path = str(meta.get("source_path") or "").strip()
            chapter = str(meta.get("chapter") or "").strip()
            section = str(meta.get("section") or "").strip()
            watch_link = str(meta.get("watch_link") or "").strip()
            source_type = scored.chunk.source_type

            url = self._source_path_to_url(source_path)
            if not url:
                url = self._watch_link_url(watch_link)

            citations.append(
                Citation(
                    chunk_id=scored.chunk.id,
                    source_path=source_path,
                    chapter=chapter,
                    section=section,
                    url=url,
                    display_label=self._build_display_label(
                        chapter,
                        section,
                        source_path,
                        source_type,
                    ),
                )
            )
        return citations
