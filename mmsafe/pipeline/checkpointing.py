"""Checkpoint and resume support for long-running evaluations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mmsafe._internal.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from mmsafe.pipeline.result_types import EvalRun

log = get_logger("pipeline.checkpoint")


def save_checkpoint(
    run: EvalRun,
    checkpoint_dir: Path,
    base_completed_ids: set[str] | None = None,
) -> Path:
    """Save current evaluation state to a checkpoint file.

    Args:
        run: Current evaluation run state.
        checkpoint_dir: Directory where checkpoint should be written.
        base_completed_ids: Optional set of completed sample IDs loaded from a
            previous checkpoint. When provided, checkpoint output remains
            cumulative-safe across resume cycles.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{run.run_id}.json"
    cumulative_completed = set(base_completed_ids or set())
    cumulative_completed.update(s.sample_id for s in run.samples)
    completed_sample_ids = sorted(cumulative_completed)

    data = {
        "run_id": run.run_id,
        "config_name": run.config_name,
        "started_at": run.started_at.isoformat(),
        "completed_sample_ids": completed_sample_ids,
        "completed_count": len(completed_sample_ids),
        "total_samples": len(run.samples),
    }

    path.write_text(json.dumps(data, indent=2))
    log.info("Checkpoint saved: %d samples → %s", len(completed_sample_ids), path)
    return path


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load completed sample IDs from a checkpoint file.

    Returns a set of sample IDs that have already been evaluated,
    so the runner can skip them on resume.
    """
    if not checkpoint_path.exists():
        return set()

    data = json.loads(checkpoint_path.read_text())
    completed = set(data.get("completed_sample_ids", []))
    log.info("Loaded checkpoint: %d completed samples from %s", len(completed), checkpoint_path)
    return completed
