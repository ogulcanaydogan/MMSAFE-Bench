"""Tests for hardware profile detection helpers."""

from __future__ import annotations

from mmsafe._internal.hardware import (
    GPUStat,
    concurrency_cap,
    resolve_hardware_profile,
    timeout_floor_seconds,
)


class TestResolveHardwareProfile:
    def test_env_override_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("MMSAFE_GPU_PROFILE", "a100")
        decision = resolve_hardware_profile("auto")
        assert decision.profile == "a100"
        assert "environment override" in decision.reason

    def test_auto_detects_idle_a100(self, monkeypatch) -> None:
        monkeypatch.delenv("MMSAFE_GPU_PROFILE", raising=False)
        monkeypatch.setattr(
            "mmsafe._internal.hardware._query_nvidia_smi",
            lambda: [
                GPUStat(
                    name="NVIDIA A100-SXM4-80GB",
                    memory_used_mb=1024,
                    memory_total_mb=81920,
                    utilization_pct=2,
                )
            ],
        )
        decision = resolve_hardware_profile("auto")
        assert decision.profile == "a100"
        assert "idle" in decision.reason.lower()

    def test_auto_falls_back_when_a100_busy(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("MMSAFE_GPU_PROFILE", raising=False)
        monkeypatch.setattr(
            "mmsafe._internal.hardware._query_nvidia_smi",
            lambda: [
                GPUStat(
                    name="NVIDIA A100-SXM4-80GB",
                    memory_used_mb=70000,
                    memory_total_mb=81920,
                    utilization_pct=95,
                )
            ],
        )
        decision = resolve_hardware_profile("auto")
        assert decision.profile == "small_gpu"
        assert "busy" in decision.reason.lower()

    def test_auto_falls_back_when_no_gpu(self, monkeypatch) -> None:
        monkeypatch.delenv("MMSAFE_GPU_PROFILE", raising=False)
        monkeypatch.setattr("mmsafe._internal.hardware._query_nvidia_smi", lambda: [])
        decision = resolve_hardware_profile("auto")
        assert decision.profile == "small_gpu"

    def test_forced_profile(self) -> None:
        decision = resolve_hardware_profile("small_gpu")
        assert decision.profile == "small_gpu"
        assert "forced by config" in decision.reason


class TestProfileTuning:
    def test_concurrency_caps(self) -> None:
        assert concurrency_cap("small_gpu") == 2
        assert concurrency_cap("a100") == 8

    def test_timeout_floors(self) -> None:
        assert timeout_floor_seconds("small_gpu") == 180
        assert timeout_floor_seconds("a100") == 120
