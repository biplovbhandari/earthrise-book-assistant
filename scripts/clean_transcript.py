"""Clean Whisper transcripts by applying text corrections.

Reads correction mappings from data/transcript_corrections.yml and applies
them to transcript segments. Logs all changes to console and a timestamped
log file under logs/.

Usage:
    uv run python scripts/clean_transcript.py [--file FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

TRANSCRIPT_DIR = Path("data/transcripts")
CORRECTIONS_FILE = Path("data/transcript_corrections.yml")
LOG_DIR = Path("logs")

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure console + timestamped file logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    handler = logging.FileHandler(LOG_DIR / f"cleanup_{timestamp}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def load_corrections(path: Path) -> dict[str, str]:
    """Load correction mappings from a YAML file."""
    if not path.exists():
        logger.warning("Corrections file not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in data.items()}


def clean_transcript(transcript_path: Path, corrections: dict[str, str]) -> list[dict]:
    """Apply corrections to transcript segments.

    Returns a list of changes, each with segment index, start time,
    original text, corrected text, and which rules matched.
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    changes: list[dict] = []
    for i, segment in enumerate(data.get("segments", [])):
        original = segment["text"]
        text = original
        applied_rules: list[str] = []
        for wrong, right in corrections.items():
            if wrong in text:
                text = text.replace(wrong, right)
                applied_rules.append(f"{wrong}->{right}")
        if text != original:
            changes.append(
                {
                    "segment": i,
                    "start": segment.get("start", 0.0),
                    "original": original.strip(),
                    "corrected": text.strip(),
                    "rules": applied_rules,
                }
            )
            segment["text"] = text

    if changes:
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changes


def main() -> int:
    """Apply corrections to one or all transcripts."""
    _setup_logging()

    parser = argparse.ArgumentParser(description="Clean Whisper transcripts.")
    parser.add_argument(
        "--file",
        type=Path,
        help="Clean a single transcript file (default: all in data/transcripts/).",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=CORRECTIONS_FILE,
        help=f"Path to corrections YAML (default: {CORRECTIONS_FILE}).",
    )
    args = parser.parse_args()

    corrections = load_corrections(args.corrections)
    if not corrections:
        logger.info("No corrections loaded, nothing to do.")
        return 0

    logger.info("Loaded %d corrections from %s", len(corrections), args.corrections)

    if args.file:
        files = [args.file]
    else:
        files = sorted(TRANSCRIPT_DIR.glob("*.json"))

    if not files:
        logger.info("No transcript files found.")
        return 0

    total_changes = 0
    for transcript_path in files:
        changes = clean_transcript(transcript_path, corrections)
        if changes:
            logger.info("%s: %d corrections applied", transcript_path.name, len(changes))
            for c in changes:
                logger.info(
                    "  CORRECTED seg %d [%.1fs]: %r -> %r (%s)",
                    c["segment"],
                    c["start"],
                    c["original"],
                    c["corrected"],
                    ", ".join(c["rules"]),
                )
            total_changes += len(changes)
        else:
            logger.info("%s: no changes", transcript_path.name)

    logger.info("=== Cleanup Complete ===")
    logger.info("Files: %d | Corrections applied: %d", len(files), total_changes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
