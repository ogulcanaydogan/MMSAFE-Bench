"""Tests for configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mmsafe.config.models import (
    AttackSpec,
    DatasetSpec,
    EdgeConfig,
    EvalConfig,
    ExecutionConfig,
    JudgeSpec,
    MetricsConfig,
    ModelSpec,
    OutputConfig,
)
from mmsafe.config.settings import Settings


class TestModelSpec:
    def test_basic_creation(self) -> None:
        spec = ModelSpec(provider="openai", model="gpt-4o", modalities=["text"])
        assert spec.provider == "openai"
        assert spec.model == "gpt-4o"
        assert spec.modalities == ["text"]
        assert spec.parameters == {}

    def test_with_parameters(self) -> None:
        spec = ModelSpec(
            provider="replicate",
            model="flux-1.1-pro",
            modalities=["image"],
            parameters={"guidance_scale": 7.5},
        )
        assert spec.parameters["guidance_scale"] == 7.5


class TestDatasetSpec:
    def test_defaults(self) -> None:
        spec = DatasetSpec(path="datasets/text/test.jsonl", modality="text")
        assert spec.is_benign is False
        assert spec.max_samples is None

    def test_benign_flag(self) -> None:
        spec = DatasetSpec(path="datasets/text/benign.jsonl", modality="text", is_benign=True)
        assert spec.is_benign is True


class TestAttackSpec:
    def test_defaults(self) -> None:
        spec = AttackSpec(name="passthrough")
        assert spec.variants == []
        assert spec.applicable_modalities is None
        assert spec.max_turns is None

    def test_with_variants(self) -> None:
        spec = AttackSpec(name="jailbreak", variants=["dan", "aim", "developer_mode"])
        assert len(spec.variants) == 3


class TestJudgeSpec:
    def test_defaults(self) -> None:
        spec = JudgeSpec(name="keyword_judge")
        assert spec.weight == 1.0
        assert spec.model is None
        assert spec.provider is None

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValidationError):
            JudgeSpec(name="test", weight=1.5)
        with pytest.raises(ValidationError):
            JudgeSpec(name="test", weight=-0.1)


class TestExecutionConfig:
    def test_defaults(self) -> None:
        config = ExecutionConfig()
        assert config.concurrency == 5
        assert config.timeout_seconds == 120
        assert config.retry_attempts == 3
        assert config.checkpoint_interval == 50
        assert config.profile == "auto"
        assert config.auto_tune is True
        assert config.strict_provider_init is False
        assert config.dry_run is False

    def test_concurrency_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(concurrency=0)
        with pytest.raises(ValidationError):
            ExecutionConfig(concurrency=101)

    def test_profile_validation(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(profile="unknown")  # type: ignore[arg-type]


class TestEdgeConfig:
    def test_defaults(self) -> None:
        config = EdgeConfig()
        assert config.enabled is False
        assert config.profile == "dgx-spark"

    def test_enabled(self) -> None:
        config = EdgeConfig(enabled=True, max_memory_gb=128, max_tokens_per_second=50)
        assert config.enabled is True
        assert config.max_memory_gb == 128


class TestMetricsConfig:
    def test_defaults(self) -> None:
        config = MetricsConfig()
        assert config.confidence_interval == 0.95
        assert config.bootstrap_samples == 1000


class TestOutputConfig:
    def test_defaults(self) -> None:
        config = OutputConfig()
        assert config.directory == "artifacts"
        assert "json" in config.formats
        assert "html" in config.formats
        assert config.leaderboard is True
        assert config.include_raw_samples is False


class TestEvalConfig:
    def test_minimal_config(self) -> None:
        config = EvalConfig(
            models=[ModelSpec(provider="stub", model="test", modalities=["text"])],
            datasets=[DatasetSpec(path="test.jsonl", modality="text")],
            judges=[JudgeSpec(name="keyword_judge")],
        )
        assert config.name == "eval-run"
        assert len(config.attacks) == 1  # default passthrough
        assert config.attacks[0].name == "passthrough"

    def test_from_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "name": "yaml-test",
            "version": "1.0",
            "models": [{"provider": "stub", "model": "test", "modalities": ["text"]}],
            "datasets": [{"path": "test.jsonl", "modality": "text"}],
            "judges": [{"name": "keyword_judge", "weight": 1.0}],
            "attacks": [{"name": "passthrough"}],
            "execution": {"concurrency": 3, "timeout_seconds": 60},
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = EvalConfig.from_yaml(yaml_path)
        assert config.name == "yaml-test"
        assert config.execution.concurrency == 3
        assert config.execution.timeout_seconds == 60
        assert len(config.models) == 1

    def test_from_yaml_with_defaults(self, tmp_path: Path) -> None:
        config_data = {
            "models": [{"provider": "stub", "model": "test", "modalities": ["text"]}],
            "datasets": [{"path": "test.jsonl", "modality": "text"}],
            "judges": [{"name": "keyword_judge"}],
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = EvalConfig.from_yaml(yaml_path)
        assert config.execution.concurrency == 5  # default
        assert config.edge.enabled is False  # default

    def test_default_configs_are_valid(self) -> None:
        defaults_dir = Path(__file__).parent.parent.parent / "mmsafe" / "config" / "defaults"
        for yaml_file in defaults_dir.glob("*.yaml"):
            config = EvalConfig.from_yaml(yaml_file)
            assert config.name
            assert len(config.models) > 0
            assert len(config.judges) > 0


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.openai_api_key == ""
        assert settings.anthropic_api_key == ""
        assert settings.mmsafe_log_level == "INFO"
        assert settings.vllm_base_url == "http://localhost:8000"
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_available_providers_no_keys(self) -> None:
        settings = Settings()
        providers = settings.available_providers()
        # Without API keys, only local backends and stub should be available
        assert "stub" in providers
        assert "local_vllm" in providers
        assert "local_ollama" in providers
        assert "openai" not in providers

    def test_available_providers_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        providers = settings.available_providers()
        assert "openai" in providers
        assert "stub" in providers
