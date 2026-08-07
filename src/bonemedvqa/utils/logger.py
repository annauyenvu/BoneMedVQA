"""Logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "bonemedvqa", level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent handlers)."""
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
