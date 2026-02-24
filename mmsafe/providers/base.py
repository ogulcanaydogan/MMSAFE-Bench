"""Abstract base interface for model providers across all modalities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mmsafe.taxonomy.categories import Modality


class ProviderStatus(str, Enum):
    """Outcome status of a generation request."""

    OK = "ok"
    REFUSED = "refused"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class GenerationRequest:
    """Unified generation request across all modalities."""

    prompt: str
    modality: Modality
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    conversation_history: tuple[dict[str, str], ...] = ()
    input_media_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResponse:
    """Unified response capturing generation output and safety signals."""

    status: ProviderStatus
    content: str | bytes | None
    content_type: str  # "text/plain", "image/png", "audio/wav", "video/mp4"
    model: str
    provider_name: str
    modality: Modality
    refusal_detected: bool = False
    refusal_reason: str = ""
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes what a provider supports."""

    name: str
    supported_modalities: tuple[Modality, ...]
    supported_models: tuple[str, ...]
    max_concurrent_requests: int = 10
    supports_streaming: bool = False
    supports_multi_turn: bool = False
    rate_limit_rpm: int | None = None


class ModelProvider(ABC):
    """Abstract interface for all generative model providers.

    Lifecycle: initialize() → generate() (many) → shutdown()
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up API clients and validate credentials."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate content from a prompt."""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Report what this provider supports."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify provider is reachable and operational."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release connections and resources."""
