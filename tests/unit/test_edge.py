"""Tests for edge deployment simulation."""

from __future__ import annotations

import pytest

from mmsafe.edge.constraints import (
    DeploymentConstraints,
    LatencyConstraints,
    ResourceConstraints,
)
from mmsafe.edge.profiles import DEVICE_PROFILES, get_profile
from mmsafe.edge.simulator import EdgeSimulator
from mmsafe.providers.base import GenerationResponse, ProviderStatus
from mmsafe.taxonomy.categories import Modality


class TestResourceConstraints:
    def test_creation(self) -> None:
        rc = ResourceConstraints(
            max_memory_gb=32,
            max_compute_tflops=125,
            max_tokens_per_second=80,
            max_batch_size=4,
            max_model_params_b=15,
            max_concurrent_requests=4,
        )
        assert rc.max_memory_gb == 32
        assert rc.max_model_params_b == 15

    def test_frozen(self) -> None:
        rc = ResourceConstraints(
            max_memory_gb=32,
            max_compute_tflops=125,
            max_tokens_per_second=80,
            max_batch_size=4,
            max_model_params_b=15,
            max_concurrent_requests=4,
        )
        with pytest.raises(AttributeError):
            rc.max_memory_gb = 64  # type: ignore[misc]


class TestDeploymentConstraints:
    def test_can_fit_small_model(self) -> None:
        profile = get_profile("dgx-spark")
        assert profile.can_fit_model(7.0, "fp16") is True

    def test_cannot_fit_huge_model_on_rpi(self) -> None:
        profile = get_profile("raspberry-pi-5")
        assert profile.can_fit_model(70.0, "fp16") is False

    def test_can_fit_quantized_model(self) -> None:
        profile = get_profile("v100-single")
        # 13B int4 ≈ 6.5GB, fits in 32GB
        assert profile.can_fit_model(13.0, "int4") is True

    def test_check_latency_pass(self) -> None:
        profile = get_profile("dgx-spark")
        passed, violations = profile.check_latency(100.0, 5000.0)
        assert passed is True
        assert violations == []

    def test_check_latency_fail(self) -> None:
        profile = get_profile("raspberry-pi-5")
        passed, violations = profile.check_latency(20000.0, 200000.0)
        assert passed is False
        assert len(violations) > 0


class TestDeviceProfiles:
    def test_all_profiles_exist(self) -> None:
        assert len(DEVICE_PROFILES) >= 5

    def test_get_profile(self) -> None:
        profile = get_profile("dgx-spark")
        assert profile.name == "dgx-spark"
        assert profile.resources.max_memory_gb == 128

    def test_get_unknown_profile(self) -> None:
        with pytest.raises(ValueError, match="Unknown device profile"):
            get_profile("nonexistent-device")

    def test_all_profiles_have_constraints(self) -> None:
        for name, profile in DEVICE_PROFILES.items():
            assert profile.name == name
            assert profile.resources.max_memory_gb > 0
            assert profile.latency.max_total_latency_ms > 0

    def test_dgx_spark_most_powerful(self) -> None:
        dgx = get_profile("dgx-spark")
        rpi = get_profile("raspberry-pi-5")
        assert dgx.resources.max_memory_gb > rpi.resources.max_memory_gb
        assert dgx.resources.max_compute_tflops > rpi.resources.max_compute_tflops


class TestEdgeSimulator:
    def test_init_with_profile_name(self) -> None:
        sim = EdgeSimulator(profile_name="dgx-spark")
        assert sim.constraints.name == "dgx-spark"

    def test_init_with_constraints(self) -> None:
        constraints = get_profile("v100-single")
        sim = EdgeSimulator(constraints=constraints)
        assert sim.constraints.name == "v100-single"

    def test_init_requires_arg(self) -> None:
        with pytest.raises(ValueError):
            EdgeSimulator()

    def test_check_model_fit_passes(self) -> None:
        sim = EdgeSimulator(profile_name="dgx-spark")
        result = sim.check_model_fit(7.0, "fp16")
        assert result.passed is True

    def test_check_model_fit_fails(self) -> None:
        sim = EdgeSimulator(profile_name="raspberry-pi-5")
        result = sim.check_model_fit(70.0, "fp16")
        assert result.passed is False
        assert any(v.constraint_type == "memory" for v in result.violations)

    def test_quantization_required(self) -> None:
        sim = EdgeSimulator(profile_name="raspberry-pi-5")
        result = sim.check_model_fit(2.0, "fp16")
        assert result.passed is False
        assert any(v.constraint_type == "quantization" for v in result.violations)

    def test_evaluate_response(self) -> None:
        sim = EdgeSimulator(profile_name="dgx-spark")
        response = GenerationResponse(
            status=ProviderStatus.OK,
            content="test",
            content_type="text/plain",
            model="test",
            provider_name="stub",
            modality=Modality.TEXT,
            latency_ms=100.0,
        )
        result = sim.evaluate_response(response)
        assert result.passed is True
        assert result.simulated_latency_ms > 0

    def test_get_summary(self) -> None:
        sim = EdgeSimulator(profile_name="dgx-spark")
        summary = sim.get_summary()
        assert summary["device_profile"] == "dgx-spark"
        assert summary["total_requests"] == 0
        assert "constraints" in summary

    def test_violation_count_tracked(self) -> None:
        sim = EdgeSimulator(profile_name="raspberry-pi-5")
        sim.check_model_fit(70.0, "fp16")
        assert sim.violation_count > 0
