"""Tests for safety metrics computation."""

from __future__ import annotations

from mmsafe.attacks.base import AttackMetadata
from mmsafe.judges.base import SafetyVerdict
from mmsafe.metrics.safety_metrics import SafetyMetrics, compute_safety_metrics
from mmsafe.pipeline.result_types import EvalSample
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import HazardCategory, Modality


def _sample(
    *,
    attack_succeeded: bool = False,
    model_refused: bool = False,
    category: HazardCategory = HazardCategory.VIOLENT_CRIME,
    attack_name: str = "passthrough",
    model: str = "gpt-4o",
    modality: Modality = Modality.TEXT,
    status: ProviderStatus = ProviderStatus.OK,
    sample_id: str = "test",
) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        prompt_id=f"p-{sample_id}",
        modality=modality,
        hazard_category=category,
        original_prompt="test prompt",
        attack_metadata=AttackMetadata(strategy_name=attack_name, original_prompt="test"),
        request=GenerationRequest(prompt="test", modality=modality, model=model),
        response=GenerationResponse(
            status=status,
            content="response",
            content_type="text/plain",
            model=model,
            provider_name="stub",
            modality=modality,
            refusal_detected=model_refused,
        ),
        verdict=SafetyVerdict(
            is_safe=not attack_succeeded,
            confidence=0.9,
            judge_name="test",
        ),
        attack_succeeded=attack_succeeded,
        model_refused=model_refused,
    )


class TestComputeSafetyMetrics:
    def test_empty_samples(self) -> None:
        metrics = compute_safety_metrics([])
        assert metrics.total_samples == 0
        assert metrics.attack_success_rate == 0.0

    def test_all_safe(self) -> None:
        samples = [_sample(sample_id=f"s{i}") for i in range(10)]
        metrics = compute_safety_metrics(samples)
        assert metrics.attack_success_rate == 0.0
        assert metrics.total_samples == 10

    def test_all_attacked(self) -> None:
        samples = [_sample(attack_succeeded=True, sample_id=f"s{i}") for i in range(10)]
        metrics = compute_safety_metrics(samples)
        assert metrics.attack_success_rate == 1.0

    def test_mixed_results(self) -> None:
        samples = [
            _sample(attack_succeeded=True, sample_id="s1"),
            _sample(attack_succeeded=False, sample_id="s2"),
            _sample(attack_succeeded=True, sample_id="s3"),
            _sample(attack_succeeded=False, sample_id="s4"),
        ]
        metrics = compute_safety_metrics(samples)
        assert metrics.attack_success_rate == 0.5

    def test_refusal_rate(self) -> None:
        samples = [
            _sample(model_refused=True, sample_id="s1"),
            _sample(model_refused=True, sample_id="s2"),
            _sample(model_refused=False, sample_id="s3"),
        ]
        metrics = compute_safety_metrics(samples)
        assert abs(metrics.refusal_rate - 2 / 3) < 0.01

    def test_asr_by_category(self) -> None:
        samples = [
            _sample(attack_succeeded=True, category=HazardCategory.VIOLENT_CRIME, sample_id="s1"),
            _sample(attack_succeeded=False, category=HazardCategory.VIOLENT_CRIME, sample_id="s2"),
            _sample(attack_succeeded=True, category=HazardCategory.HATE_SPEECH, sample_id="s3"),
        ]
        metrics = compute_safety_metrics(samples)
        assert metrics.asr_by_category["S1"] == 0.5
        assert metrics.asr_by_category["S10"] == 1.0

    def test_asr_by_attack(self) -> None:
        samples = [
            _sample(attack_succeeded=True, attack_name="jailbreak", sample_id="s1"),
            _sample(attack_succeeded=False, attack_name="passthrough", sample_id="s2"),
        ]
        metrics = compute_safety_metrics(samples)
        assert metrics.asr_by_attack["jailbreak"] == 1.0
        assert metrics.asr_by_attack["passthrough"] == 0.0

    def test_confidence_interval(self) -> None:
        samples = [
            _sample(attack_succeeded=True, sample_id=f"s{i}") for i in range(20)
        ] + [
            _sample(attack_succeeded=False, sample_id=f"f{i}") for i in range(80)
        ]
        metrics = compute_safety_metrics(samples)
        assert metrics.asr_ci_lower <= metrics.attack_success_rate
        assert metrics.asr_ci_upper >= metrics.attack_success_rate

    def test_errored_samples(self) -> None:
        samples = [
            _sample(status=ProviderStatus.ERROR, sample_id="s1"),
            _sample(status=ProviderStatus.TIMEOUT, sample_id="s2"),
            _sample(sample_id="s3"),
        ]
        metrics = compute_safety_metrics(samples)
        assert metrics.samples_errored == 2
        assert metrics.samples_evaluated == 1
