"""Search the indexed book from the command line.

Usage:
    uv run python scripts/run_query.py "What is U-Net?"
    uv run python scripts/run_query.py "What is U-Net?" --top-k 5
    uv run python scripts/run_query.py "crop mapping" --filter chapter=03_Semantic_Segmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from earthrise_rag.config import get_settings  # noqa: E402


def _parse_filters(raw: list[str] | None) -> dict[str, str] | None:
    if not raw:
        return None
    filters = {}
    for item in raw:
        if "=" not in item:
            print(f"Error: filter must be key=value, got '{item}'", file=sys.stderr)
            sys.exit(1)
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            print(f"Error: filter key and value must be non-empty, got '{item}'", file=sys.stderr)
            sys.exit(1)
        if "." in key:
            print(f"Error: filter key must not contain '.', got '{key}'", file=sys.stderr)
            sys.exit(1)
        filters[key] = value
    return filters


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Search the EarthRISE book index")
    parser.add_argument("question", help="The search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval_top_k,
        help=f"Number of results (1-50, default: {settings.retrieval_top_k})",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        help="Filter as key=value (e.g. chapter=03_Semantic_Segmentation)",
    )
    args = parser.parse_args()

    if not 1 <= args.top_k <= 50:
        print(f"Error: --top-k must be between 1 and 50, got {args.top_k}", file=sys.stderr)
        return 1

    from api.dependencies import create_pipelines

    pipelines = create_pipelines(settings)

    if (
        pipelines.query is None
        or pipelines.vector_store is None
        or pipelines.vector_store.count() == 0
    ):
        print("Error: no indexed data found. Run the indexer first.", file=sys.stderr)
        return 1

    filters = _parse_filters(args.filters)
    results = pipelines.query.search(args.question, args.top_k, filters)

    if not results:
        print("No results found.")
        return 0

    for i, scored in enumerate(results, 1):
        chunk = scored.chunk
        section = chunk.metadata.get("section", "—")
        source = chunk.metadata.get("source_path", "—")
        snippet = chunk.content[:120].replace("\n", " ")
        print(f"\n[{i}] score={scored.score:.4f}  source={source}  section={section}")
        print(f"    {snippet}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
