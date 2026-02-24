"""Result data types for evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mmsafe.attacks.base import AttackMetadata
from mmsafe.judges.base import SafetyVerdict
from mmsafe.providers.base import GenerationRequest, GenerationResponse
from mmsafe.taxonomy.categories import HazardCategory, Modality


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation sample with full provenance chain."""

    sample_id: str
    prompt_id: str
    modality: Modality
    hazard_category: HazardCategory
    original_prompt: str
    attack_metadata: AttackMetadata
    request: GenerationRequest
    response: GenerationResponse
    verdict: SafetyVerdict
    attack_succeeded: bool
    model_refused: bool


@dataclass
class EvalRun:
    """Complete evaluation run with all samples and metadata."""

    run_id: str
    config_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    samples: list[EvalSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_samples": len(self.samples),
            "metadata": self.metadata,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "prompt_id": s.prompt_id,
                    "modality": s.modality.value,
                    "hazard_category": s.hazard_category.value,
                    "original_prompt": s.original_prompt,
                    "attack_strategy": s.attack_metadata.strategy_name,
                    "attack_variant": s.attack_metadata.variant,
                    "model": s.response.model,
                    "provider": s.response.provider_name,
                    "response_status": s.response.status.value,
                    "is_safe": s.verdict.is_safe,
                    "safety_confidence": s.verdict.confidence,
                    "attack_succeeded": s.attack_succeeded,
                    "model_refused": s.model_refused,
                    "verdict_explanation": s.verdict.explanation,
                    "latency_ms": s.response.latency_ms,
                }
                for s in self.samples
            ],
        }
