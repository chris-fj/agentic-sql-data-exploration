"""Centralized logging configuration for the whole project.

Usage
-----
    from logging_config import setup_logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("…")

Uses :func:`logging.config.dictConfig` with a ``StreamHandler`` writing to
``sys.stdout`` at ``INFO`` level, so all output appears in Docker container
logs.

Coexists with uvicorn, Streamlit & friends via
``disable_existing_loggers=False`` — their own named loggers are preserved
while the root logger picks up our handler.
"""
from __future__ import annotations

import logging.config
import sys

_initialized: bool = False

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-8s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


def setup_logging() -> None:
    """Apply the shared logging configuration via :func:`logging.config.dictConfig`.

    Safe to call multiple times within the same process (subsequent calls are
    no-ops).  Uses ``disable_existing_loggers=False`` so that framework-level
    loggers configured by uvicorn or Streamlit are preserved.
    """
    global _initialized
    if _initialized:
        return
    logging.config.dictConfig(LOGGING_CONFIG)
    _initialized = True
