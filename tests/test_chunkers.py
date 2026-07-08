from earthrise_rag.indexing.chunkers import (
    NotebookChunker,
    PdfChunker,
    SectionChunker,
    VideoChunker,
)
from earthrise_rag.models import Document


def _long_content(words: int = 800) -> str:
    return ("word " * words).strip()


def _make_video_segments(count: int, duration_each: float = 10.0) -> list[dict]:
    return [
        {"start": i * duration_each, "end": (i + 1) * duration_each, "text": f"Segment {i}."}
        for i in range(count)
    ]


class TestSectionChunker:
    def test_creates_parent_child(self):
        content = f"# Title\n\nPreamble.\n\n## Section A\n\n{_long_content()}\n\n## Section B\n\n{_long_content()}"
        doc = Document(
            title="Test", source_path="book/test.md", content=content, source_type="book_text"
        )
        chunks = SectionChunker().chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_type == "child"]
        assert len(parents) >= 2
        assert len(children) >= 2
        for child in children:
            assert child.parent_id is not None
            assert any(p.id == child.parent_id for p in parents)

    def test_standalone_for_short(self):
        content = "# Title\n\n## Short Section\n\nJust a few words here."
        doc = Document(
            title="Test", source_path="book/test.md", content=content, source_type="book_text"
        )
        chunks = SectionChunker().chunk(doc)

        section_chunks = [c for c in chunks if c.metadata.get("section")]
        assert len(section_chunks) == 1
        assert all(c.chunk_type == "standalone" for c in section_chunks)
        assert all(c.parent_id is None for c in section_chunks)


class TestNotebookChunker:
    def test_groups_by_heading(self):
        doc = Document(
            title="Notebook",
            source_path="book/03/notebook.ipynb",
            content="",
            metadata={
                "cells": [
                    {"cell_type": "markdown", "source": "## Data Loading\n\nLoad the data."},
                    {"cell_type": "code", "source": "import pandas as pd"},
                    {"cell_type": "markdown", "source": "## Model Training\n\nTrain the model."},
                    {"cell_type": "code", "source": "model.fit(X, y)"},
                ],
            },
            source_type="book_text",
        )
        chunks = NotebookChunker().chunk(doc)
        sections = {c.metadata.get("section") for c in chunks if c.metadata.get("section")}
        assert "Data Loading" in sections
        assert "Model Training" in sections

    def test_long_section_creates_parent_and_code_cell_children(self):
        doc = Document(
            title="Notebook",
            source_path="book/03/notebook.ipynb",
            content="",
            metadata={
                "cells": [
                    {"cell_type": "markdown", "source": "## Big Section\n\n" + "word " * 700},
                    {"cell_type": "code", "source": "import numpy as np"},
                ],
            },
            source_type="book_text",
        )
        chunks = NotebookChunker().chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_type == "child"]
        assert len(parents) == 1
        assert parents[0].metadata.get("section") == "Big Section"
        assert children and all(c.parent_id == parents[0].id for c in children)
        code_children = [c for c in children if c.content_type == "code_cell"]
        assert len(code_children) == 1
        assert code_children[0].content == "```python\nimport numpy as np\n```"

    def test_very_large_section_splits_into_multiple_parents(self):
        doc = Document(
            title="Notebook",
            source_path="book/03/notebook.ipynb",
            content="",
            metadata={
                "cells": [
                    {"cell_type": "markdown", "source": "## Huge Section\n\nIntro."},
                    {"cell_type": "markdown", "source": "word " * 2000},
                    {"cell_type": "markdown", "source": "word " * 2000},
                ],
            },
            source_type="book_text",
        )
        chunks = NotebookChunker().chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parents) >= 2
        assert all(p.metadata.get("section") == "Huge Section" for p in parents)


class TestPdfChunker:
    def test_short_pdf_standalone(self):
        content = ("word " * 500).strip()  # 500 words < 600 standalone threshold
        doc = Document(
            title="Short Paper",
            source_path="book/ch1/pdf/short.pdf",
            content=content,
            source_type="book_text",
        )
        chunks = PdfChunker().chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "standalone"
        assert chunks[0].content_type == "paper"
        assert chunks[0].parent_id is None

    def test_long_pdf_parent_child(self):
        content = ("word " * 800).strip()  # 800 words > 600 standalone threshold
        doc = Document(
            title="Long Paper",
            source_path="book/ch1/pdf/long.pdf",
            content=content,
            source_type="book_text",
        )
        chunks = PdfChunker().chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_type == "child"]
        assert len(parents) >= 1
        assert len(children) >= 1
        assert all(c.content_type == "paper" for c in chunks)
        for child in children:
            assert child.parent_id is not None
            assert any(p.id == child.parent_id for p in parents)

    def test_empty_pdf_returns_empty(self):
        for content in ["", "   ", "\n\n"]:
            doc = Document(
                title="Empty",
                source_path="book/ch1/pdf/empty.pdf",
                content=content,
                source_type="book_text",
            )
            assert PdfChunker().chunk(doc) == []


class TestVideoChunker:
    def test_creates_parent_child_from_segments(self):
        # 30 segments x 10 s each = 300 s total, which exceeds the 240 s parent threshold.
        # The first group closes after segment 23 (end=240 s), leaving 6 segments in a second group.
        segments = _make_video_segments(30, duration_each=10.0)
        doc = Document(
            title="Lecture",
            source_path="data/transcripts/vid1.json",
            content=" ".join(s["text"] for s in segments),
            metadata={
                "video_id": "vid1",
                "url": "https://www.youtube.com/watch?v=vid1",
                "segments": segments,
            },
            source_type="video_transcript",
        )
        chapter_map = {"vid1": {"chapter": "03_Remote_Sensing", "lesson": "Intro"}}
        chunks = VideoChunker(chapter_map=chapter_map).chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_type == "child"]

        assert len(parents) == 2
        assert len(children) >= 2
        assert all(c.source_type == "video_transcript" for c in chunks)
        assert all(c.content_type == "video_segment" for c in chunks)
        assert all(c.metadata["chapter"] == "03_Remote_Sensing" for c in chunks)
        assert all(c.metadata["video_id"] == "vid1" for c in chunks)
        for child in children:
            assert child.parent_id is not None
            assert any(p.id == child.parent_id for p in parents)
        for parent in parents:
            assert "https://www.youtube.com/watch?v=vid1&t=" in parent.metadata["watch_link"]
            assert "timestamp_seconds" in parent.metadata
            assert "timestamp_end" in parent.metadata

    def test_empty_segments_returns_empty(self):
        doc = Document(
            title="Empty",
            source_path="data/transcripts/empty.json",
            content="",
            metadata={"video_id": "empty", "url": "", "segments": []},
            source_type="video_transcript",
        )
        assert VideoChunker().chunk(doc) == []

    def test_gap_forces_split(self):
        # Both segments are short (30s each), but the gap between them (500s)
        # exceeds the 240s parent threshold, so the pre-append check splits them
        # into separate parent groups.
        segments = [
            {"start": 0.0, "end": 30.0, "text": "First segment."},
            {"start": 500.0, "end": 530.0, "text": "Second segment."},
        ]
        doc = Document(
            title="Gap Test",
            source_path="data/transcripts/gap.json",
            content="First segment. Second segment.",
            metadata={"video_id": "gap", "url": "", "segments": segments},
            source_type="video_transcript",
        )
        chunks = VideoChunker().chunk(doc)

        parents = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parents) == 2
