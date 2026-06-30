from scripts.index_book import _discover_companion_pdfs, _discover_transcripts, _extract_chapters


def test_extract_chapters_from_quarto_yml():
    config = {
        "book": {
            "chapters": [
                "index.md",
                {
                    "part": "Semantic Segmentation",
                    "chapters": [
                        "03_Semantic_Segmentation/01__Crop_Mapping/notebooks/Rice_Mapping.ipynb",
                        "03_Semantic_Segmentation/02__Logging/notebooks/Detection.ipynb",
                    ],
                },
                "04_Object_Detection/index.ipynb",
            ],
        },
        "bibliography": "references.bib",
    }
    chapters = _extract_chapters(config)
    assert "index.md" in chapters
    assert "03_Semantic_Segmentation/01__Crop_Mapping/notebooks/Rice_Mapping.ipynb" in chapters
    assert "03_Semantic_Segmentation/02__Logging/notebooks/Detection.ipynb" in chapters
    assert "04_Object_Detection/index.ipynb" in chapters
    assert "references.bib" in chapters


def test_discover_companion_pdfs(tmp_path):
    chapter_dir = tmp_path / "03_Semantic_Segmentation"
    pdf_dir = chapter_dir / "pdf"
    pdf_dir.mkdir(parents=True)
    paper = pdf_dir / "companion_paper.pdf"
    paper.write_bytes(b"%PDF-1.4")

    pairs = _discover_companion_pdfs(tmp_path)

    actual_paths = [p[0] for p in pairs]
    source_paths = [p[1] for p in pairs]

    assert len(pairs) == 1
    assert str(paper) in actual_paths
    assert "book/03_Semantic_Segmentation/pdf/companion_paper.pdf" in source_paths


def test_discover_transcripts(tmp_path):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    t1 = transcript_dir / "vid1.json"
    t1.write_text('{"video_id": "vid1", "segments": []}')

    pairs = _discover_transcripts(transcript_dir)

    actual_paths = [p[0] for p in pairs]
    source_paths = [p[1] for p in pairs]

    assert len(pairs) == 1
    assert str(t1) in actual_paths
    assert "data/transcripts/vid1.json" in source_paths
