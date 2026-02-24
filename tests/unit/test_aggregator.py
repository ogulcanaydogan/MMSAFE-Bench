"""Tests for mmsafe.metrics.aggregator module."""

from __future__ import annotations

from datetime import datetime, timezone

from mmsafe.attacks.base import AttackMetadata
from mmsafe.judges.base import SafetyVerdict
from mmsafe.metrics.aggregator import (
    aggregate_by_attack,
    aggregate_by_category,
    aggregate_by_model,
    generate_summary,
)
from mmsafe.pipeline.result_types import EvalRun, EvalSample
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import HazardCategory, Modality


def _sample(
    *,
    model: str = "model-a",
    provider: str = "openai",
    attack: str = "passthrough",
    variant: str = "",
    category: HazardCategory = HazardCategory.VIOLENT_CRIME,
    attack_succeeded: bool = False,
    model_refused: bool = False,
) -> EvalSample:
    return EvalSample(
        sample_id=f"s-{model}-{attack}-{category.value}",
        prompt_id="p-001",
        modality=Modality.TEXT,
        hazard_category=category,
        original_prompt="test prompt",
        attack_metadata=AttackMetadata(
            strategy_name=attack,
            variant=variant,
            original_prompt="test prompt",
        ),
        request=GenerationRequest(prompt="test", modality=Modality.TEXT, model=model),
        response=GenerationResponse(
            status=ProviderStatus.OK,
            content="response",
            content_type="text/plain",
            model=model,
            provider_name=provider,
            modality=Modality.TEXT,
        ),
        verdict=SafetyVerdict(is_safe=not attack_succeeded, confidence=0.9, judge_name="test"),
        attack_succeeded=attack_succeeded,
        model_refused=model_refused,
    )


def _run_with_samples(samples: list[EvalSample]) -> EvalRun:
    run = EvalRun(run_id="test-run", config_name="test")
    run.samples = samples
    run.completed_at = datetime.now(timezone.utc)
    return run


class TestAggregateByModel:
    def test_groups_by_model(self) -> None:
        samples = [
            _sample(model="gpt-4o", provider="openai"),
            _sample(model="gpt-4o", provider="openai", attack_succeeded=True),
            _sample(model="claude-3", provider="anthropic"),
        ]
        result = aggregate_by_model(_run_with_samples(samples))
        assert "openai/gpt-4o" in result
        assert "anthropic/claude-3" in result
        assert result["openai/gpt-4o"].total_samples == 2
        assert result["anthropic/claude-3"].total_samples == 1


class TestAggregateByAttack:
    def test_groups_by_strategy(self) -> None:
        samples = [
            _sample(attack="passthrough"),
            _sample(attack="role_play"),
            _sample(attack="role_play"),
        ]
        result = aggregate_by_attack(_run_with_samples(samples))
        assert "passthrough" in result
        assert "role_play" in result
        assert result["role_play"].total_samples == 2

    def test_includes_variant_in_key(self) -> None:
        samples = [
            _sample(attack="role_play", variant="pirate"),
        ]
        result = aggregate_by_attack(_run_with_samples(samples))
        assert "role_play/pirate" in result


class TestAggregateByCategory:
    def test_groups_by_category(self) -> None:
        samples = [
            _sample(category=HazardCategory.VIOLENT_CRIME),
            _sample(category=HazardCategory.HATE_SPEECH),
            _sample(category=HazardCategory.HATE_SPEECH),
        ]
        result = aggregate_by_category(_run_with_samples(samples))
        assert "S1" in result
        assert "S10" in result
        assert result["S10"].total_samples == 2


class TestGenerateSummary:
    def test_complete_structure(self) -> None:
        samples = [
            _sample(attack_succeeded=True),
            _sample(model_refused=True),
            _sample(),
        ]
        summary = generate_summary(_run_with_samples(samples))
        assert "run_id" in summary
        assert "overall" in summary
        assert "by_model" in summary
        assert "by_attack" in summary
        assert "by_category" in summary
        assert "total_samples" in summary["overall"]
        assert "attack_success_rate" in summary["overall"]

    def test_empty_run(self) -> None:
        summary = generate_summary(_run_with_samples([]))
        assert summary["overall"]["total_samples"] == 0
        assert summary["overall"]["attack_success_rate"] == 0.0
