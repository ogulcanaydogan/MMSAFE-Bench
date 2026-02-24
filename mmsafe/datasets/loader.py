"""JSONL dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from mmsafe._internal.logging import get_logger

log = get_logger("datasets.loader")

_REQUIRED_FIELDS = {"id", "prompt", "category"}
_VALID_CATEGORIES = {f"S{i}" for i in range(1, 13)} | {f"X{i}" for i in range(1, 9)}
_VALID_MODALITIES = {"text", "image", "video", "audio"}


def load_dataset(path: Path, max_samples: int | None = None) -> list[dict[str, str]]:
    """Load a JSONL prompt dataset.

    Each line must be a JSON object with at least: id, prompt, category.
    Optional fields: modality, is_benign, metadata.
    """
    entries: list[dict[str, str]] = []

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            entries.append(entry)

            if max_samples and len(entries) >= max_samples:
                break

    log.info("Loaded %d samples from %s", len(entries), path.name)
    return entries


def validate_dataset(path: Path) -> list[str]:
    """Validate a JSONL dataset file and return list of errors."""
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    if not path.suffix == ".jsonl":
        errors.append(f"Expected .jsonl extension, got: {path.suffix}")

    seen_ids: set[str] = set()

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_num}: Invalid JSON: {exc}")
                continue

            if not isinstance(entry, dict):
                errors.append(f"Line {line_num}: Expected JSON object, got {type(entry).__name__}")
                continue

            # Required fields
            missing = _REQUIRED_FIELDS - set(entry.keys())
            if missing:
                errors.append(f"Line {line_num}: Missing fields: {', '.join(sorted(missing))}")
                continue

            # Unique ID
            entry_id = entry["id"]
            if entry_id in seen_ids:
                errors.append(f"Line {line_num}: Duplicate ID: {entry_id}")
            seen_ids.add(entry_id)

            # Valid category
            category = entry["category"]
            if category not in _VALID_CATEGORIES:
                errors.append(
                    f"Line {line_num}: Invalid category '{category}'. "
                    f"Must be S1-S12 or X1-X8."
                )

            # Valid modality (if present)
            if "modality" in entry and entry["modality"] not in _VALID_MODALITIES:
                errors.append(
                    f"Line {line_num}: Invalid modality '{entry['modality']}'. "
                    f"Must be one of: {', '.join(_VALID_MODALITIES)}"
                )

            # Non-empty prompt
            if not entry.get("prompt", "").strip():
                errors.append(f"Line {line_num}: Empty prompt")

    return errors
