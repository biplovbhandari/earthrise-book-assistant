from earthrise_rag.indexing.chunkers import BibChunker, NotebookChunker, SectionChunker
from earthrise_rag.models import Document


def _long_content(words: int = 800) -> str:
    return ("word " * words).strip()


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
        assert all(c.chunk_type == "standalone" for c in section_chunks)
        assert all(c.parent_id is None for c in section_chunks)

    def test_preamble_becomes_standalone(self):
        preamble_text = "This is a meaningful preamble."
        content = f"# Title\n\n{preamble_text}\n\n## Section\n\n{_long_content()}"
        doc = Document(
            title="Test", source_path="book/test.md", content=content, source_type="book_text"
        )
        chunks = SectionChunker().chunk(doc)

        preamble = [
            c for c in chunks if c.chunk_type == "standalone" and not c.metadata.get("section")
        ]
        assert len(preamble) == 1
        assert "meaningful preamble" in preamble[0].content


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


class TestBibChunker:
    def test_one_chunk_per_entry(self):
        content = "@book{key1,\n  title={Book One}\n}\n\n@article{key2,\n  title={Paper Two}\n}"
        doc = Document(
            title="References",
            source_path="book/references.bib",
            content=content,
            source_type="book_text",
        )
        chunks = BibChunker().chunk(doc)

        assert len(chunks) == 2
        assert all(c.chunk_type == "standalone" for c in chunks)
        assert all(c.content_type == "reference" for c in chunks)
        keys = {c.metadata.get("citation_key") for c in chunks}
        assert "key1" in keys
        assert "key2" in keys
