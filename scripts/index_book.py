"""Index book content into Qdrant.

Usage:
    uv run python scripts/index_book.py
    QDRANT_URL=http://localhost:6333 BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path so `api.dependencies` is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import yaml  # noqa: E402

from earthrise_rag.config import get_settings  # noqa: E402
from earthrise_rag.models.index_result import IndexResult  # noqa: E402

logger = logging.getLogger(__name__)

SKIP_FILES = {"404.qmd", "CONTRIBUTING.md", "README.md"}


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"indexing_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def _extract_chapters(quarto_config: dict) -> list[str]:
    chapters: list[str] = []
    book = quarto_config.get("book", {})

    for item in book.get("chapters", []):
        if isinstance(item, str):
            chapters.append(item)
        elif isinstance(item, dict):
            if "part" in item:
                for chapter in item.get("chapters", []):
                    if isinstance(chapter, str):
                        chapters.append(chapter)

    bib = quarto_config.get("bibliography")
    if isinstance(bib, str):
        chapters.append(bib)
    elif isinstance(bib, list):
        chapters.extend(bib)

    return chapters


def _normalize_source_path(relative_path: str, book_dir_name: str = "book") -> str:
    # Strip any leading absolute prefix (e.g. /book/) or existing book/ prefix
    clean = relative_path.lstrip("/")
    if clean.startswith(f"{book_dir_name}/"):
        return clean
    return f"{book_dir_name}/{clean}"


def main() -> int:
    _setup_logging()
    settings = get_settings()
    source_dir = Path(settings.book_source_dir)

    quarto_yml = source_dir / "_quarto.yml"
    if not quarto_yml.exists():
        logger.error("_quarto.yml not found at %s", quarto_yml)
        return 1

    try:
        with open(quarto_yml, encoding="utf-8") as f:
            quarto_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse _quarto.yml: %s", e)
        return 1

    if not isinstance(quarto_config, dict):
        logger.error("_quarto.yml is not a valid YAML mapping")
        return 1

    chapter_files = _extract_chapters(quarto_config)

    from api.dependencies import create_indexing_pipeline

    pipeline = create_indexing_pipeline(settings)

    results: list[IndexResult] = []

    for relative_path in chapter_files:
        if Path(relative_path).name in SKIP_FILES:
            continue

        actual_path = str(source_dir / relative_path)
        source_path = _normalize_source_path(relative_path)

        if not Path(actual_path).exists():
            logger.warning("File not found: %s (skipping)", actual_path)
            results.append(
                IndexResult(source_path=source_path, status="failed", error="File not found")
            )
            continue

        try:
            result = pipeline.index_source(actual_path, source_path)
            results.append(result)
            logger.info("%s: %s (%d chunks)", source_path, result.status, result.chunks_indexed)
        except Exception as e:
            logger.error("Failed to index %s: %s", source_path, e)
            results.append(IndexResult(source_path=source_path, status="failed", error=str(e)))

    # Summary
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    chunks = sum(r.chunks_indexed for r in results)

    logger.info("=== Indexing Complete ===")
    logger.info(
        "Total: %d | Success: %d | Skipped: %d | Failed: %d | Chunks: %d",
        total,
        success,
        skipped,
        failed,
        chunks,
    )

    if failed:
        for r in results:
            if r.status == "failed":
                logger.error("  FAILED: %s — %s", r.source_path, r.error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
