from earthrise_rag.indexing.chunkers import NotebookChunker, SectionChunker
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
