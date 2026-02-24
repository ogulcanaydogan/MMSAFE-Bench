"""Abstract base interface for safety evaluation judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mmsafe.providers.base import GenerationRequest, GenerationResponse
from mmsafe.taxonomy.categories import HazardCategory, Modality


@dataclass(frozen=True)
class SafetyVerdict:
    """The result of a safety evaluation on a single sample."""

    is_safe: bool
    confidence: float  # 0.0 - 1.0
    hazard_categories: tuple[HazardCategory, ...] = ()
    explanation: str = ""
    judge_name: str = ""
    raw_output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class SafetyJudge(ABC):
    """Abstract interface for safety evaluation judges."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this judge."""

    @property
    @abstractmethod
    def supported_modalities(self) -> tuple[Modality, ...]:
        """Which modalities this judge can evaluate."""

    @abstractmethod
    async def evaluate(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> SafetyVerdict:
        """Evaluate whether a model response is safe."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up judge resources."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release judge resources."""
