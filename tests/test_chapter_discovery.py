from scripts.index_book import _extract_chapters


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
