"""Abstract base interface for red-teaming attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mmsafe.providers.base import GenerationRequest
from mmsafe.taxonomy.categories import Modality


@dataclass(frozen=True)
class AttackMetadata:
    """Metadata about an applied attack transformation."""

    strategy_name: str
    variant: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    original_prompt: str = ""
    transformed_prompt: str = ""


@dataclass(frozen=True)
class AttackCapabilities:
    """Describes what an attack strategy supports."""

    name: str
    description: str
    applicable_modalities: tuple[Modality, ...]
    requires_multi_turn: bool = False


class AttackStrategy(ABC):
    """Abstract interface for red-teaming attack transformations.

    An AttackStrategy takes a GenerationRequest and returns one or more
    mutated requests designed to test safety filter robustness.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this attack strategy."""

    @abstractmethod
    def capabilities(self) -> AttackCapabilities:
        """Report what this strategy supports."""

    @abstractmethod
    def transform(
        self, request: GenerationRequest
    ) -> list[tuple[GenerationRequest, AttackMetadata]]:
        """Transform a request into one or more attack variants.

        Returns list of (mutated_request, metadata) tuples.
        """
