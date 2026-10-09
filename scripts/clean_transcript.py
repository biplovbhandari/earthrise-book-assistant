"""Clean Whisper transcripts by removing artifacts and applying corrections.

Four cleanup categories, applied in order:
1. Hallucination removal - fake "subscribe"/"thanks for watching" segments
   that Whisper injects mid-lecture.
2. Garbled segment removal - single characters, non-Latin script in English
   transcripts, and other gibberish from audio processing failures.
3. Duplicate removal - consecutive segments with identical text, a common
   Whisper artifact when confidence is low.
4. Term corrections - dictionary-based text replacements from
   data/transcript_corrections.yml.

Logs all changes to console and a timestamped log file under logs/.

Usage:
    uv run python scripts/clean_transcript.py [--file FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import yaml

TRANSCRIPT_DIR = Path("data/transcripts")
CORRECTIONS_FILE = Path("data/transcript_corrections.yml")
LOG_DIR = Path("logs")

logger = logging.getLogger(__name__)

_HALLUCINATION_PATTERNS = [
    r"thank(?:s| you) for watching",
    r"please (?:like|subscribe)",
    r"like and subscribe",
    r"don'?t forget to subscribe",
    r"see you (?:in the )?next",
    r"if you enjoyed this video",
    r"hit the (?:like|bell|notification)",
    r"leave a comment below",
]
_HALLUCINATION_RE = re.compile("|".join(_HALLUCINATION_PATTERNS), re.IGNORECASE)

_MIN_SEGMENT_LENGTH = 3

_HALLUCINATION_TAIL_SKIP = 10


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


def _is_hallucination(text: str, index: int, total: int) -> bool:
    """Check if segment text matches common Whisper hallucination patterns.

    Skips the last few segments where these phrases may be legitimate
    speaker sign-offs rather than Whisper artifacts.
    """
    if index >= total - _HALLUCINATION_TAIL_SKIP:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return _HALLUCINATION_RE.search(stripped) is not None


def _is_garbled(text: str) -> bool:
    """Check if segment text is garbled or gibberish."""
    stripped = text.strip()
    if len(stripped) < _MIN_SEGMENT_LENGTH:
        return True
    latin_count = sum(
        1
        for c in stripped
        if unicodedata.category(c).startswith("L")
        and ord(c) < 0x0250  # Basic Latin + Latin Extended-A
    )
    letter_count = sum(1 for c in stripped if unicodedata.category(c).startswith("L"))
    return letter_count > 0 and latin_count / letter_count < 0.5


def _is_duplicate(text: str, prev_text: str | None) -> bool:
    """Check if segment text is identical to the previous segment."""
    if prev_text is None:
        return False
    return text.strip() == prev_text.strip()


def clean_transcript(transcript_path: Path, corrections: dict[str, str]) -> dict[str, list[dict]]:
    """Apply all cleanup categories to a transcript.

    Returns a dict keyed by category with lists of changes/removals.
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    original_count = len(segments)

    removals: dict[str, list[dict]] = {
        "hallucination": [],
        "garbled": [],
        "duplicate": [],
        "correction": [],
    }

    indices_to_remove: set[int] = set()

    # Pass 1: Mark hallucinations
    for i, seg in enumerate(segments):
        if _is_hallucination(seg["text"], i, original_count):
            removals["hallucination"].append(
                {
                    "segment": i,
                    "start": seg.get("start", 0.0),
                    "text": seg["text"].strip(),
                }
            )
            indices_to_remove.add(i)

    # Pass 2: Mark garbled segments
    for i, seg in enumerate(segments):
        if i in indices_to_remove:
            continue
        if _is_garbled(seg["text"]):
            removals["garbled"].append(
                {
                    "segment": i,
                    "start": seg.get("start", 0.0),
                    "text": seg["text"].strip(),
                }
            )
            indices_to_remove.add(i)

    # Pass 3: Mark duplicates (consecutive identical text, skipping already-removed)
    prev_text: str | None = None
    for i, seg in enumerate(segments):
        if i in indices_to_remove:
            continue
        if _is_duplicate(seg["text"], prev_text):
            removals["duplicate"].append(
                {
                    "segment": i,
                    "start": seg.get("start", 0.0),
                    "text": seg["text"].strip(),
                }
            )
            indices_to_remove.add(i)
        else:
            prev_text = seg["text"]

    # Remove marked segments
    if indices_to_remove:
        data["segments"] = [seg for i, seg in enumerate(segments) if i not in indices_to_remove]
        segments = data["segments"]

    # Pass 4: Apply corrections on remaining segments
    for i, seg in enumerate(segments):
        original = seg["text"]
        text = original
        applied_rules: list[str] = []
        for wrong, right in corrections.items():
            if wrong in text:
                text = text.replace(wrong, right)
                applied_rules.append(f"{wrong}->{right}")
        if text != original:
            removals["correction"].append(
                {
                    "segment": i,
                    "start": seg.get("start", 0.0),
                    "original": original.strip(),
                    "corrected": text.strip(),
                    "rules": applied_rules,
                }
            )
            seg["text"] = text

    has_changes = any(removals[k] for k in removals)
    if has_changes:
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    final_count = len(data["segments"])
    removed_count = original_count - final_count

    return {
        **removals,
        "_summary": [
            {
                "original_segments": original_count,
                "final_segments": final_count,
                "removed": removed_count,
            }
        ],
    }


def _log_results(filename: str, results: dict[str, list[dict]]) -> int:
    """Log cleanup results for one file. Returns total change count."""
    summary = results["_summary"][0]
    total = 0

    for category in ("hallucination", "garbled", "duplicate"):
        items = results[category]
        if items:
            total += len(items)
            logger.info("%s: %d %s(s) removed", filename, len(items), category)
            for item in items:
                logger.info(
                    "  REMOVED seg %d [%.1fs] (%s): %r",
                    item["segment"],
                    item["start"],
                    category,
                    item["text"],
                )

    corrections = results["correction"]
    if corrections:
        total += len(corrections)
        logger.info("%s: %d correction(s) applied", filename, len(corrections))
        for c in corrections:
            logger.info(
                "  CORRECTED seg %d [%.1fs]: %r -> %r (%s)",
                c["segment"],
                c["start"],
                c["original"],
                c["corrected"],
                ", ".join(c["rules"]),
            )

    if total == 0:
        logger.info("%s: no changes", filename)
    elif summary["removed"] > 0:
        logger.info(
            "%s: %d -> %d segments (%d removed)",
            filename,
            summary["original_segments"],
            summary["final_segments"],
            summary["removed"],
        )

    return total


def main() -> int:
    """Apply cleanup to one or all transcripts."""
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
        results = clean_transcript(transcript_path, corrections)
        total_changes += _log_results(transcript_path.name, results)

    logger.info("=== Cleanup Complete ===")
    logger.info("Files: %d | Total changes: %d", len(files), total_changes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
