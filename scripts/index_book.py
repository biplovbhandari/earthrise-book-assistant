"""Index book content into Qdrant.

Discovers chapters from the book's Quarto config (_quarto.yml, merged with the
profile file _quarto-<profile>.yml when one exists). Also indexes companion PDFs
under book/<chapter>/pdf/ and video transcripts from data/transcripts/.

Usage:
    uv run python scripts/index_book.py
    QDRANT_URL=http://localhost:6333 BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on sys.path so `api.dependencies` is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import yaml

from earthrise_rag.config import get_settings
from earthrise_rag.models.index_result import IndexResult

logger = logging.getLogger(__name__)

SKIP_FILES = {"404.qmd", "CONTRIBUTING.md", "README.md"}
TRANSCRIPT_DIR = Path("data/transcripts")
LOG_DIR = Path("logs")


def _setup_logging() -> None:
    """Configure console + timestamped file logging under logs/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"indexing_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def _load_quarto_config(source_dir: Path) -> dict | None:
    """Load and merge Quarto config files.

    Quarto splits config across _quarto.yml (base) and profile files like
    _quarto-book.yml. The profile file carries the chapter list when the base
    file uses `profile: default: book`. We merge them so the indexer sees
    the same chapter list Quarto does.
    """
    base_path = source_dir / "_quarto.yml"
    if not base_path.exists():
        logger.error("_quarto.yml not found at %s", base_path)
        return None

    try:
        with open(base_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse _quarto.yml: %s", e)
        return None

    if not isinstance(config, dict):
        logger.error("_quarto.yml is not a valid YAML mapping")
        return None

    profile_name = config.get("profile", {}).get("default", "")
    if profile_name:
        profile_path = source_dir / f"_quarto-{profile_name}.yml"
        if profile_path.exists():
            try:
                with open(profile_path, encoding="utf-8") as f:
                    profile_config = yaml.safe_load(f) or {}
                for key, value in profile_config.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        config[key].update(value)
                    else:
                        config[key] = value
            except yaml.YAMLError as e:
                logger.warning(
                    "Failed to parse %s: %s (using base config only)", profile_path.name, e
                )

    return config


def _extract_chapters(quarto_config: dict) -> list[str]:
    """Extract chapter file paths from a parsed _quarto.yml config.

    Handles both flat chapter lists and part-grouped chapters.
    Also appends bibliography entries if present.
    """
    chapters: list[str] = []
    book = quarto_config.get("book", {})

    for item in book.get("chapters", []):
        if isinstance(item, str):
            chapters.append(item)
        elif isinstance(item, dict) and "part" in item:
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
    """Ensure source_path always starts with ``book/`` for consistent metadata."""
    # Strip any leading absolute prefix (e.g. /book/) or existing book/ prefix
    clean = relative_path.lstrip("/")
    if clean.startswith(f"{book_dir_name}/"):
        return clean
    return f"{book_dir_name}/{clean}"


def _discover_companion_pdfs(source_dir: Path) -> list[tuple[str, str]]:
    """Return (actual_path, source_path) pairs for all PDFs under book/<chapter>/pdf/."""
    pairs: list[tuple[str, str]] = []
    for pdf_path in source_dir.rglob("pdf/*.pdf"):
        relative = pdf_path.relative_to(source_dir)
        source_path = _normalize_source_path(str(relative))
        pairs.append((str(pdf_path), source_path))
    return sorted(pairs)


def _discover_transcripts(transcript_dir: Path) -> list[tuple[str, str]]:
    """Return (actual_path, source_path) pairs for all JSON transcripts.

    Returns an empty list when transcript_dir does not exist so the indexing
    script runs cleanly before any transcripts have been generated.
    source_path is fixed to data/transcripts/<filename> and is NOT derived
    from book_source_dir.
    """
    if not transcript_dir.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for json_path in transcript_dir.glob("*.json"):
        source_path = f"data/transcripts/{json_path.name}"
        pairs.append((str(json_path), source_path))
    return sorted(pairs)


def main() -> int:
    """Index book chapters, companion PDFs, and transcripts into Qdrant."""
    _setup_logging()

    parser = argparse.ArgumentParser(description="Index book content into Qdrant.")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    source_dir = Path(settings.book_source_dir)

    quarto_config = _load_quarto_config(source_dir)
    if quarto_config is None:
        return 1

    chapter_files = _extract_chapters(quarto_config)

    if args.recreate_collection:
        from uuid import uuid4

        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            Modifier,
            PointStruct,
            SparseIndexParams,
            SparseVector,
            SparseVectorParams,
            VectorParams,
        )

        client = QdrantClient(url=settings.qdrant_url)

        temp_collection = f"{settings.qdrant_collection}_preflight_{uuid4().hex[:8]}"
        created_temp = False

        # Prove the full sparse path works (model load, Qdrant schema, upsert)
        # before deleting the real collection (destructive). If any step fails
        # the old dense index stays intact.
        try:
            from api.dependencies import _create_sparse_embedder

            sparse_embedder = _create_sparse_embedder(settings)
            test_emb = sparse_embedder.embed(["preflight test"])[0]

            sparse_params = SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF if sparse_embedder.requires_idf else None,
            )
            client.create_collection(
                collection_name=temp_collection,
                vectors_config={"dense": VectorParams(size=10, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": sparse_params},
            )
            created_temp = True

            client.upsert(
                collection_name=temp_collection,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector={
                            "dense": [0.0] * 10,
                            "sparse": SparseVector(
                                indices=list(test_emb.indices),
                                values=list(test_emb.values),
                            ),
                        },
                        payload={},
                    )
                ],
            )
            logger.info("Sparse preflight OK")
        except Exception as e:
            logger.error("Sparse preflight failed: %s. Collection NOT deleted.", e)
            return 1
        finally:
            if created_temp:
                try:
                    client.delete_collection(temp_collection)
                except Exception:
                    logger.warning("Failed to clean up preflight collection '%s'", temp_collection)

        if settings.qdrant_collection in [c.name for c in client.get_collections().collections]:
            client.delete_collection(settings.qdrant_collection)
            logger.info("Deleted collection '%s'", settings.qdrant_collection)
        else:
            logger.info(
                "Collection '%s' does not exist, nothing to delete", settings.qdrant_collection
            )

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

    companion_pdfs = _discover_companion_pdfs(source_dir)
    for actual_path, source_path in companion_pdfs:
        try:
            result = pipeline.index_source(actual_path, source_path)
            results.append(result)
            logger.info("%s: %s (%d chunks)", source_path, result.status, result.chunks_indexed)
        except Exception as e:
            logger.error("Failed to index %s: %s", source_path, e)
            results.append(IndexResult(source_path=source_path, status="failed", error=str(e)))

    transcripts = _discover_transcripts(TRANSCRIPT_DIR)
    for actual_path, source_path in transcripts:
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
