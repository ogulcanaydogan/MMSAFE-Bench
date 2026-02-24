"""Metric aggregation across categories, models, and attacks."""

from __future__ import annotations

from typing import Any

from mmsafe.metrics.safety_metrics import SafetyMetrics, compute_safety_metrics
from mmsafe.pipeline.result_types import EvalRun, EvalSample


def aggregate_by_model(run: EvalRun) -> dict[str, SafetyMetrics]:
    """Compute safety metrics broken down by model.

    Args:
        run: Completed evaluation run containing all samples.

    Returns:
        Mapping of ``provider/model`` keys to their computed SafetyMetrics.
    """
    groups: dict[str, list[EvalSample]] = {}
    for sample in run.samples:
        key = f"{sample.response.provider_name}/{sample.response.model}"
        groups.setdefault(key, []).append(sample)

    return {model: compute_safety_metrics(samples) for model, samples in groups.items()}


def aggregate_by_attack(run: EvalRun) -> dict[str, SafetyMetrics]:
    """Compute safety metrics broken down by attack strategy.

    Args:
        run: Completed evaluation run containing all samples.

    Returns:
        Mapping of attack strategy names (with optional variant suffix) to metrics.
    """
    groups: dict[str, list[EvalSample]] = {}
    for sample in run.samples:
        key = sample.attack_metadata.strategy_name
        if sample.attack_metadata.variant:
            key = f"{key}/{sample.attack_metadata.variant}"
        groups.setdefault(key, []).append(sample)

    return {attack: compute_safety_metrics(samples) for attack, samples in groups.items()}


def aggregate_by_category(run: EvalRun) -> dict[str, SafetyMetrics]:
    """Compute safety metrics broken down by hazard category.

    Args:
        run: Completed evaluation run containing all samples.

    Returns:
        Mapping of hazard category codes (e.g. ``S1``, ``X3``) to metrics.
    """
    groups: dict[str, list[EvalSample]] = {}
    for sample in run.samples:
        key = sample.hazard_category.value
        groups.setdefault(key, []).append(sample)

    return {cat: compute_safety_metrics(samples) for cat, samples in groups.items()}


def generate_summary(run: EvalRun) -> dict[str, Any]:
    """Generate a comprehensive summary of the evaluation run.

    Computes overall metrics and per-model, per-attack, and per-category
    breakdowns. The returned dictionary is JSON-serializable and intended
    to be stored alongside raw results.

    Args:
        run: Completed evaluation run containing all samples.

    Returns:
        Summary dictionary with ``overall``, ``by_model``, ``by_attack``,
        and ``by_category`` keys.
    """
    overall = compute_safety_metrics(run.samples)
    by_model = aggregate_by_model(run)
    by_attack = aggregate_by_attack(run)
    by_category = aggregate_by_category(run)

    return {
        "run_id": run.run_id,
        "config_name": run.config_name,
        "duration_seconds": run.duration_seconds,
        "overall": {
            "total_samples": overall.total_samples,
            "attack_success_rate": round(overall.attack_success_rate, 4),
            "refusal_rate": round(overall.refusal_rate, 4),
            "false_refusal_rate": round(overall.false_refusal_rate, 4),
            "asr_ci": [round(overall.asr_ci_lower, 4), round(overall.asr_ci_upper, 4)],
            "samples_errored": overall.samples_errored,
        },
        "by_model": {
            model: {
                "asr": round(m.attack_success_rate, 4),
                "rr": round(m.refusal_rate, 4),
                "samples": m.total_samples,
            }
            for model, m in by_model.items()
        },
        "by_attack": {
            attack: {
                "asr": round(m.attack_success_rate, 4),
                "samples": m.total_samples,
            }
            for attack, m in by_attack.items()
        },
        "by_category": {
            cat: {
                "asr": round(m.attack_success_rate, 4),
                "rr": round(m.refusal_rate, 4),
                "samples": m.total_samples,
            }
            for cat, m in by_category.items()
        },
    }
