from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.document import Document

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^##\s+", re.MULTILINE)
_WORD_LIMIT_CHILD = 500
_STANDALONE_THRESHOLD = 600
_NOTEBOOK_PARENT_CAP = 3000


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _word_count(text: str) -> int:
    return len(text.split())


def _chapter_from_path(source_path: str) -> str:
    parts = source_path.split("/")
    for part in parts:
        if part and part[0].isdigit():
            return part
    return ""


def _make_chunk(
    *,
    content: str,
    source_path: str,
    content_type: str,
    chunk_type: str,
    parent_id: str | None,
    commit_sha: str,
    extra_metadata: dict[str, Any] | None = None,
    section: str = "",
) -> Chunk:
    metadata = {
        "chapter": _chapter_from_path(source_path),
        "section": section,
        "source_path": source_path,
        "commit_sha": commit_sha,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Chunk(
        content=content,
        content_hash=_content_hash(content),
        source_type="book_text",
        content_type=content_type,
        metadata=metadata,
        chunk_type=chunk_type,
        parent_id=parent_id,
    )


def _split_paragraphs(text: str, target_words: int) -> list[str]:
    paragraphs = text.split("\n\n")
    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = _word_count(para)

        if current_words + para_words > target_words and current:
            segments.append("\n\n".join(current))
            current = []
            current_words = 0

        if para_words > target_words:
            words = para.split()
            for i in range(0, len(words), target_words):
                segments.append(" ".join(words[i : i + target_words]))
        else:
            current.append(para)
            current_words += para_words

    if current:
        segments.append("\n\n".join(current))

    return segments


class SectionChunker:
    """Splits markdown by ## headings into parent/child or standalone chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        commit_sha = document.metadata.get("commit_sha", "")
        source_path = document.source_path

        splits = _SECTION_RE.split(document.content)
        heading_matches = list(re.finditer(r"^##\s+(.+)", document.content, re.MULTILINE))

        preamble = splits[0].strip() if splits else ""
        heading_names = [m.group(1).strip() for m in heading_matches]
        sections = (
            list(zip(heading_names, [s.strip() for s in splits[1:]])) if len(splits) > 1 else []
        )

        if preamble:
            logger.info("  Preamble: %d words → standalone", _word_count(preamble))
            chunks.append(
                _make_chunk(
                    content=preamble,
                    source_path=source_path,
                    content_type="concept",
                    chunk_type="standalone",
                    parent_id=None,
                    commit_sha=commit_sha,
                )
            )

        for section_name, section_text in sections:
            # section_text starts with "Section Name\n\nBody..."
            # Strip the heading line to get just the body for child splitting
            _, _, section_body = section_text.partition("\n")
            section_body = section_body.lstrip("\n")
            full_section = f"## {section_text}"
            word_count = _word_count(full_section)

            if word_count < _STANDALONE_THRESHOLD:
                logger.info("  Section '%s': %d words → standalone", section_name, word_count)
                chunks.append(
                    _make_chunk(
                        content=full_section,
                        source_path=source_path,
                        content_type="concept",
                        chunk_type="standalone",
                        parent_id=None,
                        commit_sha=commit_sha,
                        section=section_name,
                    )
                )
            else:
                parent = _make_chunk(
                    content=full_section,
                    source_path=source_path,
                    content_type="concept",
                    chunk_type="parent",
                    parent_id=None,
                    commit_sha=commit_sha,
                    section=section_name,
                )
                chunks.append(parent)

                child_segments = _split_paragraphs(section_body, _WORD_LIMIT_CHILD)
                logger.info(
                    "  Section '%s': %d words → parent + %d children",
                    section_name,
                    word_count,
                    len(child_segments),
                )
                for segment in child_segments:
                    if segment.strip():
                        chunks.append(
                            _make_chunk(
                                content=segment,
                                source_path=source_path,
                                content_type="concept",
                                chunk_type="child",
                                parent_id=parent.id,
                                commit_sha=commit_sha,
                                section=section_name,
                            )
                        )

        logger.info("  SectionChunker: %d chunks from %s", len(chunks), source_path)
        return chunks


class NotebookChunker:
    """Chunks notebooks by grouping cells under ## headings."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        commit_sha = document.metadata.get("commit_sha", "")
        source_path = document.source_path
        cells = document.metadata.get("cells", [])

        groups = self._group_by_heading(cells)
        logger.info("  NotebookChunker: %d heading groups in %s", len(groups), source_path)

        for section_name, group_cells in groups:
            content_parts = []
            for cell in group_cells:
                if cell["cell_type"] == "code":
                    content_parts.append(f"```python\n{cell['source']}\n```")
                else:
                    content_parts.append(cell["source"])

            full_content = "\n\n".join(content_parts)
            word_count = _word_count(full_content)

            if not section_name:
                if full_content.strip():
                    logger.info("  Preamble: %d words → standalone", word_count)
                    chunks.append(
                        _make_chunk(
                            content=full_content,
                            source_path=source_path,
                            content_type="concept",
                            chunk_type="standalone",
                            parent_id=None,
                            commit_sha=commit_sha,
                        )
                    )
                continue

            if word_count < _STANDALONE_THRESHOLD:
                logger.info("  Section '%s': %d words → standalone", section_name, word_count)
                chunks.append(
                    _make_chunk(
                        content=full_content,
                        source_path=source_path,
                        content_type="concept",
                        chunk_type="standalone",
                        parent_id=None,
                        commit_sha=commit_sha,
                        section=section_name,
                    )
                )
                continue

            if word_count > _NOTEBOOK_PARENT_CAP:
                sub_groups = self._split_large_group(group_cells, _NOTEBOOK_PARENT_CAP)
                logger.info(
                    "  Section '%s': %d words → split into %d sub-parents",
                    section_name,
                    word_count,
                    len(sub_groups),
                )
                for sub_cells in sub_groups:
                    sub_parts = []
                    for cell in sub_cells:
                        if cell["cell_type"] == "code":
                            sub_parts.append(f"```python\n{cell['source']}\n```")
                        else:
                            sub_parts.append(cell["source"])
                    sub_content = "\n\n".join(sub_parts)
                    parent = _make_chunk(
                        content=sub_content,
                        source_path=source_path,
                        content_type="concept",
                        chunk_type="parent",
                        parent_id=None,
                        commit_sha=commit_sha,
                        section=section_name,
                    )
                    chunks.append(parent)
                    self._add_child_chunks(
                        sub_cells, chunks, source_path, parent.id, commit_sha, section_name
                    )
            else:
                logger.info(
                    "  Section '%s': %d words → parent + %d cell children",
                    section_name,
                    word_count,
                    len(group_cells),
                )
                parent = _make_chunk(
                    content=full_content,
                    source_path=source_path,
                    content_type="concept",
                    chunk_type="parent",
                    parent_id=None,
                    commit_sha=commit_sha,
                    section=section_name,
                )
                chunks.append(parent)
                self._add_child_chunks(
                    group_cells, chunks, source_path, parent.id, commit_sha, section_name
                )

        logger.info("  NotebookChunker: %d chunks from %s", len(chunks), source_path)
        return chunks

    def _group_by_heading(self, cells: list[dict]) -> list[tuple[str, list[dict]]]:
        groups: list[tuple[str, list[dict]]] = []
        current_heading = ""
        current_cells: list[dict] = []

        for cell in cells:
            if cell["cell_type"] == "markdown":
                match = re.search(r"^##\s+(.+)", cell["source"], re.MULTILINE)
                if match:
                    if current_cells:
                        groups.append((current_heading, current_cells))
                    current_heading = match.group(1).strip()
                    current_cells = [cell]
                    continue
            current_cells.append(cell)

        if current_cells:
            groups.append((current_heading, current_cells))

        return groups

    def _add_child_chunks(
        self,
        cells: list[dict],
        chunks: list[Chunk],
        source_path: str,
        parent_id: str,
        commit_sha: str,
        section_name: str,
    ) -> None:
        for cell in cells:
            ct = "code_cell" if cell["cell_type"] == "code" else "concept"
            content = cell["source"]
            if cell["cell_type"] == "code":
                content = f"```python\n{content}\n```"
            if content.strip():
                chunks.append(
                    _make_chunk(
                        content=content,
                        source_path=source_path,
                        content_type=ct,
                        chunk_type="child",
                        parent_id=parent_id,
                        commit_sha=commit_sha,
                        section=section_name,
                    )
                )

    def _split_large_group(self, cells: list[dict], max_words: int) -> list[list[dict]]:
        sub_groups: list[list[dict]] = []
        current: list[dict] = []
        current_words = 0

        for cell in cells:
            cell_words = _word_count(cell["source"])
            if current_words + cell_words > max_words and current:
                sub_groups.append(current)
                current = [cell]
                current_words = cell_words
            else:
                current.append(cell)
                current_words += cell_words

        if current:
            sub_groups.append(current)

        return sub_groups


class BibChunker:
    """Splits BibTeX content into one standalone chunk per entry."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        commit_sha = document.metadata.get("commit_sha", "")
        source_path = document.source_path

        entries = re.split(r"(?=^@)", document.content, flags=re.MULTILINE)
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            citation_key = ""
            key_match = re.match(r"@\w+\{([\w\-.:]+),", entry)
            if key_match:
                citation_key = key_match.group(1)

            logger.info("  BibEntry: %s", citation_key)
            chunks.append(
                _make_chunk(
                    content=entry,
                    source_path=source_path,
                    content_type="reference",
                    chunk_type="standalone",
                    parent_id=None,
                    commit_sha=commit_sha,
                    extra_metadata={"citation_key": citation_key},
                )
            )

        logger.info("  BibChunker: %d entries from %s", len(chunks), source_path)
        return chunks
