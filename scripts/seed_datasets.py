#!/usr/bin/env python3
"""Generate seed prompt datasets for all modalities.

This script validates existing datasets and provides statistics.
Run with --regenerate to recreate from scratch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def summarize_datasets() -> None:
    """Print summary statistics for all datasets."""
    print("MMSAFE-Bench Dataset Summary")
    print("=" * 60)

    total_prompts = 0
    for modality_dir in sorted(DATASETS_DIR.iterdir()):
        if not modality_dir.is_dir():
            continue

        modality_total = 0
        print(f"\n{modality_dir.name}/")

        for jsonl_file in sorted(modality_dir.glob("*.jsonl")):
            count = sum(1 for line in jsonl_file.read_text().strip().split("\n") if line.strip())
            modality_total += count

            # Count categories
            categories: dict[str, int] = {}
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                cat = entry.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(categories.items()))
            print(f"  {jsonl_file.name:40s}  {count:3d} prompts  [{cat_str}]")

        total_prompts += modality_total
        print(f"  {'Subtotal':40s}  {modality_total:3d}")

    print(f"\n{'=' * 60}")
    print(f"Total prompts across all datasets: {total_prompts}")


def main() -> int:
    if not DATASETS_DIR.exists():
        print(f"Datasets directory not found: {DATASETS_DIR}")
        return 1

    summarize_datasets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
