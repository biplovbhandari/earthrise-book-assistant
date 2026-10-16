"""Structured logging configuration for the application."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

request_id: ContextVar[str] = ContextVar("request_id", default="")

APP_LOG_DIR = Path("logs/app")

_JSON_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_JSON_RENAME = {"asctime": "timestamp", "levelname": "level", "name": "logger"}


class RequestIDFilter(logging.Filter):
    """Inject the current request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()  # type: ignore[attr-defined]
        return True


def configure_logging(*, json: bool = True, level: str = "INFO") -> None:
    """Configure the root logger with console and file output.

    Console uses JSON in production, plain text in dev.
    File always uses JSON for machine parseability, rotated at 10 MB.

    Args:
        json: Use JSON formatter for console (production). False uses human-readable (dev).
        level: Root log level.
    """
    root = logging.getLogger()
    root.setLevel(level)

    root.handlers.clear()

    console = logging.StreamHandler()
    if json:
        console.setFormatter(JsonFormatter(fmt=_JSON_FMT, rename_fields=_JSON_RENAME))
    else:
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    APP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        APP_LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(JsonFormatter(fmt=_JSON_FMT, rename_fields=_JSON_RENAME))
    root.addHandler(file_handler)

    root.addFilter(RequestIDFilter())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
