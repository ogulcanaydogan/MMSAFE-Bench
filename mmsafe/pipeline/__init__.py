"""Evaluation pipeline exports.

Keep imports light at module import time to avoid circular imports between
pipeline runner and reporting/metrics modules.
"""

from __future__ import annotations

from mmsafe.pipeline.result_types import EvalRun, EvalSample

__all__ = ["EvalRun", "EvalRunner", "EvalSample"]


def __getattr__(name: str) -> object:
    """Lazy-load heavy symbols to prevent import cycles."""
    if name == "EvalRunner":
        from mmsafe.pipeline.runner import EvalRunner

        return EvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
