"""Content hashing utilities for reproducibility and deduplication."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_content(content: str | bytes) -> str:
    """Return a SHA-256 hex digest of the given content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_dict(data: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hash of a dictionary.

    Keys are sorted for reproducibility.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hash_content(serialized)


def short_id(content: str | bytes, length: int = 8) -> str:
    """Return a short hash ID suitable for display."""
    return hash_content(content)[:length]
