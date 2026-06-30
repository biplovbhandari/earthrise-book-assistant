import json

import nbformat
import pytest
from unittest.mock import MagicMock, patch

from earthrise_rag.indexing.parsers import (
    MarkdownParser,
    NotebookParser,
    PdfParser,
    TranscriptParser,
)


class TestMarkdownParser:
    def test_extracts_title_and_strips_frontmatter(self, tmp_path):
        md_file = tmp_path / "chapter.md"
        md_file.write_text(
            "# Introduction\n\nThis is the introduction.\n\n## Section 1\n\nContent here."
        )
        doc = MarkdownParser().parse(str(md_file), "book/chapter.md")
        assert doc.title == "Introduction"
        assert doc.source_path == "book/chapter.md"
        assert "This is the introduction." in doc.content
        assert doc.source_type == "book_text"

        qmd_file = tmp_path / "page.qmd"
        qmd_file.write_text('---\ntitle: "How to Cite"\n---\n\n# Citation Guide\n\nCite this book.')
        doc2 = MarkdownParser().parse(str(qmd_file), "book/citing.qmd")
        assert doc2.title == "How to Cite"
        assert "---" not in doc2.content
        assert "Cite this book." in doc2.content


class TestNotebookParser:
    def test_preserves_cell_structure(self, tmp_path):
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# Chapter Title\n\nIntro paragraph."),
            nbformat.v4.new_code_cell("import numpy as np"),
            nbformat.v4.new_markdown_cell("## Section 1\n\nMore content."),
        ]
        nb_file = tmp_path / "chapter.ipynb"
        nbformat.write(nb, str(nb_file))

        doc = NotebookParser().parse(str(nb_file), "book/03/chapter.ipynb")
        assert doc.title == "Chapter Title"
        assert "cells" in doc.metadata
        assert len(doc.metadata["cells"]) == 3
        assert doc.metadata["cells"][0]["cell_type"] == "markdown"
        assert doc.metadata["cells"][1]["cell_type"] == "code"


class TestPdfParser:
    def test_extracts_title_from_metadata(self, tmp_path):
        pdf_path = str(tmp_path / "paper.pdf")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page one content."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page two content."

        mock_pdf = MagicMock()
        mock_pdf.metadata = {"Title": "Research Paper"}
        mock_pdf.pages = [mock_page1, mock_page2]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_open.return_value.__exit__.return_value = False

            doc = PdfParser().parse(pdf_path, "book/ch1/pdf/paper.pdf")

        assert doc.title == "Research Paper"
        assert doc.source_path == "book/ch1/pdf/paper.pdf"
        assert "Page one content." in doc.content
        assert "Page two content." in doc.content
        assert doc.source_type == "book_text"
        assert doc.metadata["page_count"] == 2

    def test_title_falls_back_to_stem(self, tmp_path):
        pdf_path = str(tmp_path / "my_research_paper.pdf")

        mock_pdf = MagicMock()
        mock_pdf.metadata = {}
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some content."
        mock_pdf.pages = [mock_page]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_open.return_value.__exit__.return_value = False

            doc = PdfParser().parse(pdf_path, "book/ch1/pdf/my_research_paper.pdf")

        assert doc.title == "my research paper"

    def test_skips_none_page_text(self, tmp_path):
        pdf_path = str(tmp_path / "paper.pdf")

        mock_page_none = MagicMock()
        mock_page_none.extract_text.return_value = None
        mock_page_real = MagicMock()
        mock_page_real.extract_text.return_value = "Real content."

        mock_pdf = MagicMock()
        mock_pdf.metadata = {"Title": "Test"}
        mock_pdf.pages = [mock_page_none, mock_page_real]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_open.return_value.__exit__.return_value = False

            doc = PdfParser().parse(pdf_path, "book/ch1/pdf/paper.pdf")

        assert doc.content == "Real content."
        # page_count reflects all pages regardless of whether text was extracted
        assert doc.metadata["page_count"] == 2


class TestTranscriptParser:
    def test_parses_valid_transcript(self, tmp_path):
        data = {
            "video_id": "abc123",
            "title": "Introduction to Remote Sensing",
            "url": "https://www.youtube.com/watch?v=abc123",
            "duration_seconds": 600.0,
            "segments": [
                {"start": 0.0, "end": 10.0, "text": "Hello world."},
                {"start": 10.0, "end": 20.0, "text": "Welcome to the course."},
            ],
        }
        transcript_file = tmp_path / "abc123.json"
        transcript_file.write_text(json.dumps(data))

        doc = TranscriptParser().parse(str(transcript_file), "data/transcripts/abc123.json")

        assert doc.title == "Introduction to Remote Sensing"
        assert doc.source_path == "data/transcripts/abc123.json"
        assert doc.source_type == "video_transcript"
        assert doc.content == "Hello world. Welcome to the course."
        assert doc.metadata["video_id"] == "abc123"
        assert doc.metadata["url"] == "https://www.youtube.com/watch?v=abc123"
        assert doc.metadata["duration_seconds"] == 600.0
        assert len(doc.metadata["segments"]) == 2

    def test_missing_segment_fields_raises(self, tmp_path):
        data = {
            "video_id": "bad1",
            "segments": [{"start": 0.0, "end": 5.0}],
        }
        transcript_file = tmp_path / "bad.json"
        transcript_file.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="text"):
            TranscriptParser().parse(str(transcript_file), "data/transcripts/bad.json")
