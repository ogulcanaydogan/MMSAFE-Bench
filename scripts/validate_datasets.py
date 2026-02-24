#!/usr/bin/env python3
"""Validate all JSONL datasets in the repository."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mmsafe.datasets.loader import validate_dataset


def main() -> int:
    datasets_dir = PROJECT_ROOT / "datasets"
    if not datasets_dir.exists():
        print(f"Datasets directory not found: {datasets_dir}")
        return 1

    jsonl_files = sorted(datasets_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return 1

    print(f"Validating {len(jsonl_files)} dataset files...\n")

    total_errors = 0
    for jsonl_file in jsonl_files:
        rel_path = jsonl_file.relative_to(PROJECT_ROOT)
        errors = validate_dataset(jsonl_file)
        if errors:
            print(f"  FAIL  {rel_path}")
            for err in errors:
                print(f"        - {err}")
            total_errors += len(errors)
        else:
            print(f"  OK    {rel_path}")

    print(f"\n{'=' * 60}")
    if total_errors:
        print(f"FAILED: {total_errors} error(s) found")
        return 1

    print(f"PASSED: All {len(jsonl_files)} datasets valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
