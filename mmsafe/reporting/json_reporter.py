"""JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmsafe.metrics.aggregator import generate_summary
from mmsafe.pipeline.result_types import EvalRun


def generate_json_report(run: EvalRun, output_path: Path) -> Path:
    """Generate a comprehensive JSON report from an evaluation run.

    Returns the path to the written report.
    """
    report: dict[str, Any] = {
        "version": "1.0",
        "framework": "mmsafe-bench",
        **run.to_dict(),
        "summary": generate_summary(run),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return output_path


def load_json_report(path: Path) -> dict[str, Any]:
    """Load a JSON report from disk."""
    return json.loads(path.read_text())
