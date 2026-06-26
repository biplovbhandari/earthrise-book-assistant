from __future__ import annotations

import logging
import re

from earthrise_rag.models.document import Document

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class MarkdownParser:
    """Parses .md and .qmd files, stripping anchored YAML frontmatter."""

    def parse(self, actual_path: str, source_path: str) -> Document:
        text = _read_text(actual_path)
        metadata: dict = {}
        title = ""

        match = _FRONTMATTER_RE.match(text)
        if match:
            try:
                import yaml

                fm = yaml.safe_load(match.group(1))
                if isinstance(fm, dict):
                    title = fm.get("title", "")
                    metadata["frontmatter"] = fm
            except Exception as e:
                logger.warning("Failed to parse YAML frontmatter in %s: %s", actual_path, e)
            text = text[match.end() :]

        if not title:
            heading = _HEADING_RE.search(text)
            if heading:
                title = heading.group(1).strip()

        return Document(
            title=title or source_path,
            source_path=source_path,
            content=text,
            metadata=metadata,
            source_type="book_text",
        )


class NotebookParser:
    """Parses .ipynb files, preserving cell structure in metadata."""

    def parse(self, actual_path: str, source_path: str) -> Document:
        import nbformat

        nb = nbformat.read(actual_path, as_version=4)
        metadata: dict = {}
        cells_data: list[dict] = []
        content_parts: list[str] = []
        title = ""

        for cell in nb.cells:
            cell_type = cell.cell_type
            source = cell.source

            if cell_type == "raw" and source.strip().startswith("---"):
                try:
                    import yaml

                    raw_text = source.strip()
                    if raw_text.startswith("---"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("---"):
                        raw_text = raw_text[:-3]
                    fm = yaml.safe_load(raw_text)
                    if isinstance(fm, dict):
                        metadata.update(fm)
                except Exception as e:
                    logger.warning("Failed to parse raw cell YAML in %s: %s", actual_path, e)
                continue

            cells_data.append({"cell_type": cell_type, "source": source})

            if cell_type == "markdown":
                content_parts.append(source)
                if not title:
                    heading = _HEADING_RE.search(source)
                    if heading:
                        title = heading.group(1).strip()
            elif cell_type == "code":
                content_parts.append(f"```python\n{source}\n```")

        metadata["cells"] = cells_data

        return Document(
            title=title or source_path,
            source_path=source_path,
            content="\n\n".join(content_parts),
            metadata=metadata,
            source_type="book_text",
        )


class BibParser:
    """Parses .bib files as plain text for splitting by BibChunker."""

    def parse(self, actual_path: str, source_path: str) -> Document:
        text = _read_text(actual_path)
        return Document(
            title="References",
            source_path=source_path,
            content=text,
            metadata={},
            source_type="book_text",
        )
