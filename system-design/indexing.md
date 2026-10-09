# Indexing Pipeline

How the RAG assistant discovers, parses, and indexes book content into Qdrant.

## Content Discovery

The indexer (`scripts/index_book.py`) discovers content from three sources:

```
Source 1: Book chapters
  Config:    _quarto.yml + _quarto-<profile>.yml (merged)
  Discovery: Explicit file paths under book.chapters
  Types:     .ipynb (NotebookParser), .md/.qmd (MarkdownParser), .bib (BibParser)

Source 2: Companion PDFs
  Config:    None (convention-based)
  Discovery: rglob("pdf/*.pdf") under book source directory
  Types:     .pdf (PdfParser)

Source 3: Video transcripts
  Config:    data/video_chapter_map.yml (maps video IDs to chapters)
  Discovery: All .json files in data/transcripts/
  Types:     .json (TranscriptParser)
```

## Quarto Config Contract

The indexer reads the book's Quarto configuration to find chapter file paths.
Quarto supports profile-based config splitting:

- `_quarto.yml` contains base settings (title, format, bibliography)
- `_quarto-<profile>.yml` (e.g., `_quarto-book.yml`) contains the chapter listing

When `_quarto.yml` declares `profile: default: book`, the indexer merges
`_quarto-book.yml` into the base config.
This mirrors what Quarto does at render time.

If the book repo changes how profiles are named or structured, the indexer's
`_load_quarto_config()` function must be updated to match.

## File Type to Parser/Chunker Mapping

| Extension | Parser | Chunker | Content |
|-----------|--------|---------|---------|
| `.md`, `.qmd` | MarkdownParser | SectionChunker | Book chapter text |
| `.ipynb` | NotebookParser | NotebookChunker | Notebooks with code + markdown cells |
| `.pdf` | PdfParser | PdfChunker | Research papers, companion materials |
| `.json` | TranscriptParser | VideoChunker | YouTube lecture transcripts |
| `.bib` | BibParser | BibChunker | Bibliography entries |

## Video Chapter Map

`data/video_chapter_map.yml` maps YouTube video IDs to book chapter directories.
The indexer uses this to attach chapter metadata to transcript chunks for citations.

```yaml
videos:
  <video_id>:
    chapter: "<directory name>"    # e.g., "10_Future"
    lesson: "<subdirectory name>"  # e.g., "02__Comprehensive_EOFM_Benchmarking"
```

When a new lecture video is published:
1. Add its ID to `video_chapter_map.yml`
2. Transcribe: `just transcribe <video_id>`
3. Re-index: `just index` (or `--recreate-collection` for a full rebuild)

## What Is NOT Indexed

- Images (PNG, JPG) - text captions in markdown are indexed, not the images themselves
- Data files (CSV, TIFF, model weights)
- Presentation files (PPTX) - convert to PDF first, place in `pdf/` directory
- Draft chapters not listed in the Quarto config
