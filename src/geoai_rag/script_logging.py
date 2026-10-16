"""Shared logging setup for CLI scripts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

SCRIPT_LOG_DIR = Path("logs/scripts")


def _prune_old_logs(prefix: str, keep: int = 10) -> None:
    """Remove old log files, keeping the most recent *keep* per prefix."""
    if not SCRIPT_LOG_DIR.exists():
        return
    files = sorted(SCRIPT_LOG_DIR.glob(f"{prefix}_*.log"), reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def setup_script_logging(prefix: str) -> None:
    """Configure console + timestamped JSON file logging under logs/scripts/.

    Args:
        prefix: Log filename prefix (e.g. "indexing", "cleanup", "transcription").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(SCRIPT_LOG_DIR / f"{prefix}_{timestamp}.log")
    file_handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
    )
    logging.getLogger().addHandler(file_handler)

    _prune_old_logs(prefix)
