"""Shared logging setup for CLI scripts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

LOG_DIR = Path("logs")


def setup_script_logging(prefix: str) -> None:
    """Configure console + timestamped file logging under logs/.

    Args:
        prefix: Log filename prefix (e.g. "indexing", "cleanup", "transcription").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"{prefix}_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
