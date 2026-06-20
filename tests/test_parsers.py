import nbformat

from earthrise_rag.indexing.parsers import BibParser, MarkdownParser, NotebookParser


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

    def test_preserves_midfile_horizontal_rules(self, tmp_path):
        md_file = tmp_path / "page.md"
        md_file.write_text("# Title\n\nBefore rule.\n\n---\n\nAfter rule.")
        doc = MarkdownParser().parse(str(md_file), "book/page.md")
        assert "Before rule." in doc.content
        assert "After rule." in doc.content
        assert "---" in doc.content


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


class TestBibParser:
    def test_returns_content(self, tmp_path):
        bib_file = tmp_path / "references.bib"
        bib_file.write_text(
            "@book{key1,\n  title={Book One}\n}\n\n@article{key2,\n  title={Paper Two}\n}"
        )
        doc = BibParser().parse(str(bib_file), "book/references.bib")
        assert doc.title == "References"
        assert "@book{key1" in doc.content
        assert "@article{key2" in doc.content
