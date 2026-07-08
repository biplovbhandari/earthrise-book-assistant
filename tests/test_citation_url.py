from typing import Literal

from earthrise_rag.citations.citation_builder import DefaultCitationBuilder
from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.scored_chunk import ScoredChunk

_url = DefaultCitationBuilder._source_path_to_url
_humanize = DefaultCitationBuilder._humanize_dir_name
_filename = DefaultCitationBuilder._filename_label
_label = DefaultCitationBuilder._build_display_label
_watch = DefaultCitationBuilder._watch_link_url


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

    def test_double_slash_in_path_rejected(self):
        # Isolates the "//" clause: after stripping "book/", path is "a//b.qmd"
        # with no leading slash, so it fires the "//" in path guard, not startswith("/").
        assert _url("book/a//b.qmd") == ""

    def test_backslash_rejected(self):
        assert _url("book/foo\\..\\secret.qmd") == ""

    def test_traversal_rejected(self):
        assert _url("book/../etc/passwd.qmd") == ""

    def test_control_chars_rejected(self):
        assert _url("book/foo\x00bar.qmd") == ""

    def test_query_string_encoded(self):
        assert _url("book/a?b=c.qmd") == "/a%3Fb%3Dc.html"


class TestHumanizeDirName:
    def test_chapter_with_single_underscore(self):
        assert _humanize("03_Semantic_Segmentation") == "Semantic Segmentation"

    def test_lesson_with_double_underscore(self):
        assert _humanize("01__Active_Fire_Detection") == "Active Fire Detection"

    def test_empty_string(self):
        assert _humanize("") == ""

    def test_no_digit_prefix(self):
        assert _humanize("citing") == "citing"


class TestFilenameLabel:
    def test_qmd(self):
        assert _filename("book/citing.qmd") == "Citing"

    def test_pdf(self):
        assert _filename("book/06_Eco/pdf/active_fire.pdf") == "Active Fire"

    def test_empty(self):
        assert _filename("") == ""


class TestBuildDisplayLabel:
    def test_video_transcript_both_humanized(self):
        result = _label(
            "06_Eco_Process_Sim",
            "01__Active_Fire_Detection",
            "data/transcripts/vid.json",
            "video_transcript",
        )
        assert result == "Eco Process Sim - Active Fire Detection"

    def test_book_text_preserves_underscores_in_heading(self):
        result = _label(
            "03_Semantic_Segmentation",
            "API_Response",
            "book/03_Seg/index.qmd",
            "book_text",
        )
        assert result == "Semantic Segmentation - API_Response"

    def test_pdf_chapter_with_filename(self):
        result = _label(
            "06_Eco_Process_Sim",
            "",
            "book/06_Eco/pdf/active_fire.pdf",
            "book_text",
        )
        assert result == "Eco Process Sim - Active Fire"

    def test_section_only(self):
        assert _label("", "U-Net", "book/ch.qmd", "book_text") == "U-Net"

    def test_filename_fallback(self):
        assert _label("", "", "book/citing.qmd", "book_text") == "Citing"

    def test_all_empty_returns_source(self):
        assert _label("", "", "", "book_text") == "Source"


class TestWatchLinkUrl:
    def test_valid_with_timestamp(self):
        link = "https://www.youtube.com/watch?v=e6lqjTUAvmI&t=45s"
        assert _watch(link) == link

    def test_valid_without_timestamp(self):
        link = "https://www.youtube.com/watch?v=e6lqjTUAvmI"
        assert _watch(link) == link

    def test_empty_string(self):
        assert _watch("") == ""

    def test_non_youtube_url(self):
        assert _watch("https://evil.com/watch?v=abc") == ""

    def test_malformed_host(self):
        assert _watch("https://youtube.com/watch?v=abc") == ""

    def test_empty_video_id(self):
        assert _watch("https://www.youtube.com/watch?v=&t=0s") == ""

    def test_trailing_newline_rejected(self):
        assert _watch("https://www.youtube.com/watch?v=abc\n") == ""

    def test_extra_query_params_rejected(self):
        assert _watch("https://www.youtube.com/watch?v=abc&t=5s&list=PL") == ""


def _make_chunk(
    *, source_type: Literal["book_text", "video_transcript"] = "book_text", metadata=None
):
    """Build a Chunk directly (not via conftest helper) for integration tests."""
    return Chunk(
        content="test content",
        content_hash="abc123",
        source_type=source_type,
        content_type="concept",
        metadata=metadata or {},
    )


class TestBuildIntegration:
    def test_video_chunk_gets_watch_link_and_label(self):
        chunk = _make_chunk(
            source_type="video_transcript",
            metadata={
                "source_path": "data/transcripts/e6lqjTUAvmI.json",
                "chapter": "06_Eco_Process_Sim",
                "section": "01__Active_Fire_Detection",
                "watch_link": "https://www.youtube.com/watch?v=e6lqjTUAvmI&t=45s",
            },
        )
        scored = ScoredChunk(chunk=chunk, score=0.9, ranking_method="dense")
        citations = DefaultCitationBuilder().build([scored])
        assert citations[0].url == "https://www.youtube.com/watch?v=e6lqjTUAvmI&t=45s"
        assert citations[0].display_label == "Eco Process Sim - Active Fire Detection"

    def test_book_chunk_uses_source_path_url(self):
        chunk = _make_chunk(
            metadata={
                "source_path": "book/citing.qmd",
                "chapter": "",
                "section": "",
            },
        )
        scored = ScoredChunk(chunk=chunk, score=0.8, ranking_method="dense")
        citations = DefaultCitationBuilder().build([scored])
        assert citations[0].url == "/citing.html"
        assert citations[0].display_label == "Citing"

    def test_source_path_url_takes_precedence_over_watch_link(self):
        chunk = _make_chunk(
            metadata={
                "source_path": "book/03_Seg/index.qmd",
                "chapter": "03_Seg",
                "section": "U-Net",
                "watch_link": "https://www.youtube.com/watch?v=abc&t=10s",
            },
        )
        scored = ScoredChunk(chunk=chunk, score=0.9, ranking_method="dense")
        citations = DefaultCitationBuilder().build([scored])
        assert citations[0].url == "/03_Seg/index.html"

    def test_unmapped_transcript_gets_filename_label(self):
        chunk = _make_chunk(
            source_type="video_transcript",
            metadata={
                "source_path": "data/transcripts/unknown_vid.json",
                "chapter": "",
                "section": "",
                "watch_link": "https://www.youtube.com/watch?v=unknown_vid&t=0s",
            },
        )
        scored = ScoredChunk(chunk=chunk, score=0.7, ranking_method="dense")
        citations = DefaultCitationBuilder().build([scored])
        assert citations[0].url == "https://www.youtube.com/watch?v=unknown_vid&t=0s"
        assert citations[0].display_label == "Unknown Vid"

    def test_two_transcript_chunks_same_source_different_timestamps(self):
        meta_base = {
            "source_path": "data/transcripts/vid1.json",
            "chapter": "03_Semantic_Segmentation",
            "section": "01__Crop_Mapping",
        }
        chunk1 = _make_chunk(
            source_type="video_transcript",
            metadata={**meta_base, "watch_link": "https://www.youtube.com/watch?v=vid1&t=45s"},
        )
        chunk2 = _make_chunk(
            source_type="video_transcript",
            metadata={**meta_base, "watch_link": "https://www.youtube.com/watch?v=vid1&t=300s"},
        )
        scored = [
            ScoredChunk(chunk=chunk1, score=0.9, ranking_method="dense"),
            ScoredChunk(chunk=chunk2, score=0.8, ranking_method="dense"),
        ]
        citations = DefaultCitationBuilder().build(scored)
        assert citations[0].url != citations[1].url
        assert "&t=45s" in citations[0].url
        assert "&t=300s" in citations[1].url
        assert citations[0].display_label == "Semantic Segmentation - Crop Mapping"
        assert citations[1].display_label == "Semantic Segmentation - Crop Mapping"


class TestMetadataCoercion:
    def test_non_string_metadata_does_not_crash(self):
        chunk = _make_chunk(
            metadata={
                "source_path": 123,
                "chapter": None,
                "section": 7,
                "watch_link": None,
            },
        )
        scored = ScoredChunk(chunk=chunk, score=0.5, ranking_method="dense")
        citations = DefaultCitationBuilder().build([scored])
        c = citations[0]
        assert c.source_path == "123"
        assert c.chapter == ""
        assert c.section == "7"
        assert c.url == ""
        assert c.display_label == "7"
