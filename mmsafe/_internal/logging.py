"""Structured logging setup for MMSAFE-Bench."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the root mmsafe logger with rich formatting."""
    logger = logging.getLogger("mmsafe")

    if logger.handlers:
        return logger

    handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the mmsafe namespace."""
    return logging.getLogger(f"mmsafe.{name}")


# Ensure stderr for logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
