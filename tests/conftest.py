"""Shared test fixtures for MMSAFE-Bench."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmsafe.config.models import (
    AttackSpec,
    DatasetSpec,
    EvalConfig,
    ExecutionConfig,
    JudgeSpec,
    MetricsConfig,
    ModelSpec,
    OutputConfig,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def sample_model_spec() -> ModelSpec:
    return ModelSpec(
        provider="stub",
        model="stub-text-v1",
        modalities=["text"],
    )


@pytest.fixture
def sample_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        path="datasets/text/mlcommons_hazards.jsonl",
        modality="text",
    )


@pytest.fixture
def sample_benign_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        path="datasets/text/benign_prompts.jsonl",
        modality="text",
        is_benign=True,
    )


@pytest.fixture
def sample_attack_spec() -> AttackSpec:
    return AttackSpec(name="passthrough")


@pytest.fixture
def sample_judge_spec() -> JudgeSpec:
    return JudgeSpec(
        name="keyword_judge",
        weight=1.0,
    )


@pytest.fixture
def sample_eval_config(
    sample_model_spec: ModelSpec,
    sample_dataset_spec: DatasetSpec,
    sample_attack_spec: AttackSpec,
    sample_judge_spec: JudgeSpec,
) -> EvalConfig:
    return EvalConfig(
        name="test-eval",
        version="1.0",
        models=[sample_model_spec],
        datasets=[sample_dataset_spec],
        attacks=[sample_attack_spec],
        judges=[sample_judge_spec],
        execution=ExecutionConfig(
            concurrency=2,
            timeout_seconds=30,
            retry_attempts=1,
            checkpoint_interval=10,
            dry_run=False,
        ),
        metrics=MetricsConfig(
            confidence_interval=0.95,
            bootstrap_samples=100,
        ),
        output=OutputConfig(
            directory="artifacts/test",
            formats=["json"],
            leaderboard=False,
        ),
    )


@pytest.fixture
def sample_config_yaml(tmp_path: Path, sample_eval_config: EvalConfig) -> Path:
    """Write a sample config to a temporary YAML file."""
    import yaml

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        yaml.dump(sample_eval_config.model_dump(), default_flow_style=False)
    )
    return config_path
