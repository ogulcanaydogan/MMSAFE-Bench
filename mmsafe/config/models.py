"""Pydantic configuration models for MMSAFE-Bench evaluation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class ModelSpec(BaseModel):
    """Specification for a model to evaluate."""

    provider: str = Field(description="Provider name: openai, anthropic, google, replicate, etc.")
    model: str = Field(description="Model identifier within the provider")
    modalities: list[str] = Field(description="Modalities this model should be tested with")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific parameters"
    )


class DatasetSpec(BaseModel):
    """Specification for a prompt dataset."""

    path: str = Field(description="Path to JSONL dataset file")
    modality: str = Field(description="Dataset modality: text, image, video, audio")
    is_benign: bool = Field(
        default=False, description="Whether this is a benign dataset for FRR measurement"
    )
    max_samples: int | None = Field(
        default=None, description="Limit number of samples loaded from this dataset"
    )


class AttackSpec(BaseModel):
    """Specification for an attack strategy."""

    name: str = Field(description="Attack strategy name")
    variants: list[str] = Field(default_factory=list, description="Specific variants to apply")
    applicable_modalities: list[str] | None = Field(
        default=None, description="Restrict attack to these modalities"
    )
    max_turns: int | None = Field(default=None, description="Max turns for multi-turn attacks")
    escalation: str | None = Field(default=None, description="Escalation strategy name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )


class JudgeSpec(BaseModel):
    """Specification for a safety judge."""

    name: str = Field(description="Judge implementation name")
    model: str | None = Field(default=None, description="Model to use for LLM-based judges")
    provider: str | None = Field(default=None, description="Provider for judge model")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight in composite scoring")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Judge-specific parameters"
    )


class ExecutionConfig(BaseModel):
    """Execution engine parameters."""

    concurrency: int = Field(default=5, ge=1, le=100, description="Max concurrent API calls")
    timeout_seconds: int = Field(default=120, ge=10, le=600, description="Per-request timeout")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Retry count on failure")
    checkpoint_interval: int = Field(
        default=50, ge=1, description="Save checkpoint every N samples"
    )
    profile: Literal["auto", "small_gpu", "a100"] = Field(
        default="auto",
        description="Execution profile. auto detects hardware and falls back safely.",
    )
    auto_tune: bool = Field(
        default=True,
        description="Apply profile-based runtime tuning for safer cross-GPU configs.",
    )
    strict_provider_init: bool = Field(
        default=False,
        description="Fail fast when any configured provider cannot initialize.",
    )
    dry_run: bool = Field(default=False, description="Validate config without running")


class EdgeConfig(BaseModel):
    """Edge deployment simulation parameters."""

    enabled: bool = Field(default=False, description="Enable edge simulation")
    profile: str = Field(default="dgx-spark", description="Device profile name")
    max_memory_gb: float | None = Field(
        default=None, description="Memory constraint in GB"
    )
    max_tokens_per_second: float | None = Field(
        default=None, description="Throughput constraint"
    )


class MetricsConfig(BaseModel):
    """Metrics computation parameters."""

    confidence_interval: float = Field(
        default=0.95, ge=0.8, le=0.99, description="CI level for bootstrap"
    )
    bootstrap_samples: int = Field(
        default=1000, ge=100, description="Number of bootstrap resamples"
    )


class OutputConfig(BaseModel):
    """Report output parameters."""

    directory: str = Field(default="artifacts", description="Output directory for results")
    formats: list[str] = Field(
        default_factory=lambda: ["json", "html", "markdown"],
        description="Report output formats",
    )
    leaderboard: bool = Field(default=True, description="Generate comparative leaderboard")
    include_raw_samples: bool = Field(
        default=False, description="Include raw model outputs in results (privacy risk)"
    )


class EvalConfig(BaseModel):
    """Top-level evaluation configuration loaded from YAML."""

    name: str = Field(default="eval-run", description="Human-readable name for this evaluation")
    version: str = Field(default="1.0", description="Config schema version")
    models: list[ModelSpec] = Field(description="Models to evaluate")
    datasets: list[DatasetSpec] = Field(description="Prompt datasets")
    attacks: list[AttackSpec] = Field(
        default_factory=lambda: [AttackSpec(name="passthrough")],
        description="Attack strategies to apply",
    )
    judges: list[JudgeSpec] = Field(description="Safety judges for evaluation")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        """Load and validate an evaluation config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
