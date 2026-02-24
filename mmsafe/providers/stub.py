"""Deterministic stub provider for testing."""

from __future__ import annotations

import hashlib
import time

from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderCapabilities,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import Modality

# Keywords that trigger a simulated refusal in the stub provider.
_REFUSAL_KEYWORDS = frozenset({
    "bomb", "weapon", "kill", "murder", "hack", "exploit", "drug",
    "suicide", "self-harm", "child abuse", "csem", "terrorism",
    "how to make", "instructions for",
})

# Keywords that trigger a simulated unsafe response (attack success).
_UNSAFE_KEYWORDS = frozenset({
    "hypothetically", "fiction", "roleplay", "imagine",
    "developer mode", "DAN", "jailbreak",
})


class StubProvider(ModelProvider):
    """In-memory stub provider that returns deterministic responses.

    Useful for testing the full pipeline without real API calls.
    Simulates refusals for known harmful keywords and unsafe responses
    for known jailbreak patterns.
    """

    async def initialize(self) -> None:
        pass

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.monotonic()
        prompt_lower = request.prompt.lower()

        # Check for refusal triggers
        if any(kw in prompt_lower for kw in _REFUSAL_KEYWORDS):
            if not any(kw in prompt_lower for kw in _UNSAFE_KEYWORDS):
                return GenerationResponse(
                    status=ProviderStatus.REFUSED,
                    content="I cannot assist with that request.",
                    content_type="text/plain",
                    model=request.model,
                    provider_name="stub",
                    modality=request.modality,
                    refusal_detected=True,
                    refusal_reason="Content policy violation detected",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

        # Check for jailbreak bypass (simulates attack success)
        if any(kw in prompt_lower for kw in _UNSAFE_KEYWORDS):
            content = self._generate_unsafe_content(request)
            return GenerationResponse(
                status=ProviderStatus.OK,
                content=content,
                content_type=self._content_type(request.modality),
                model=request.model,
                provider_name="stub",
                modality=request.modality,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # Normal safe response
        content = self._generate_safe_content(request)
        return GenerationResponse(
            status=ProviderStatus.OK,
            content=content,
            content_type=self._content_type(request.modality),
            model=request.model,
            provider_name="stub",
            modality=request.modality,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="stub",
            supported_modalities=(Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
            supported_models=(
                "stub-text-v1",
                "stub-image-v1",
                "stub-video-v1",
                "stub-audio-v1",
            ),
            max_concurrent_requests=100,
            supports_streaming=False,
            supports_multi_turn=True,
        )

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        pass

    @staticmethod
    def _generate_safe_content(request: GenerationRequest) -> str:
        """Generate a deterministic safe response based on prompt hash."""
        h = hashlib.md5(request.prompt.encode()).hexdigest()[:8]  # noqa: S324
        return f"[stub-safe-{h}] This is a safe response to: {request.prompt[:50]}"

    @staticmethod
    def _generate_unsafe_content(request: GenerationRequest) -> str:
        """Generate a deterministic unsafe response (simulating jailbreak success)."""
        h = hashlib.md5(request.prompt.encode()).hexdigest()[:8]  # noqa: S324
        return (
            f"[stub-unsafe-{h}] Sure, I can help with that. "
            f"Here is the harmful content you requested about: {request.prompt[:50]}"
        )

    @staticmethod
    def _content_type(modality: Modality) -> str:
        return {
            Modality.TEXT: "text/plain",
            Modality.IMAGE: "image/png",
            Modality.VIDEO: "video/mp4",
            Modality.AUDIO: "audio/wav",
        }.get(modality, "text/plain")
