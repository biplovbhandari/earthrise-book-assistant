from earthrise_rag.citations.citation_builder import DefaultCitationBuilder

_url = DefaultCitationBuilder._source_path_to_url


class TestSourcePathToUrl:
    def test_ipynb_to_html(self):
        source = "book/03_Semantic_Segmentation/01__Crop_Mapping/notebooks/Rice_Mapping_Bhutan_2021.ipynb"
        expected = (
            "/03_Semantic_Segmentation/01__Crop_Mapping/notebooks/Rice_Mapping_Bhutan_2021.html"
        )
        assert _url(source) == expected

    def test_qmd_to_html(self):
        assert _url("book/citing.qmd") == "/citing.html"

    def test_md_to_html(self):
        assert _url("book/index.md") == "/index.html"

    def test_empty_string(self):
        assert _url("") == ""

    def test_non_book_prefix(self):
        assert _url("data/transcripts/foo.json") == ""

    def test_pdf_not_rendered(self):
        assert _url("book/03_Segmentation/pdf/file.pdf") == ""

    def test_protocol_relative_rejected(self):
        assert _url("book//evil.example/a.qmd") == ""

    def test_backslash_rejected(self):
        assert _url("book/foo\\..\\secret.qmd") == ""

    def test_traversal_rejected(self):
        assert _url("book/../etc/passwd.qmd") == ""

    def test_control_chars_rejected(self):
        assert _url("book/foo\x00bar.qmd") == ""

    def test_query_string_encoded(self):
        assert _url("book/a?b=c.qmd") == "/a%3Fb%3Dc.html"
