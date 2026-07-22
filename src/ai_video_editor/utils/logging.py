"""Shared logging configuration for the AI Video Editor pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _PROJECT_ROOT / "logs"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stderr and logs/<root>.log.

    Safe to call repeatedly for the same name; handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    top_level = name.split(".")[0]
    file_handler = logging.FileHandler(_LOG_DIR / f"{top_level}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
