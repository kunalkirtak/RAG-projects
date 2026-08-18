"""
logger.py
---------
Centralised logging setup. Every module in the project should call
`get_logger(__name__)` rather than instantiating its own logger, so that
formatting, log level, and log file destination stay consistent.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

from src.config import config

_CONFIGURED_LOGGERS = set()


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured logger that writes to console (rich, if available)
    and to a timestamped file under `logs/`.

    Args:
        name: Usually `__name__` of the calling module.
        log_file: Optional override for the log filename.

    Returns:
        A ready-to-use `logging.Logger`.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if _HAS_RICH:
        console_handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    else:  # pragma: no cover
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_dir = Path(config.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_name = log_file or f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(logs_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
