"""Replicate provider for multi-modal generation (text, image, video, audio)."""

from __future__ import annotations

import time
from typing import Any

from mmsafe._internal.logging import get_logger
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderCapabilities,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import Modality

log = get_logger("providers.replicate")

_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "cannot assist",
    "content policy",
    "nsfw",
    "not allowed",
    "violates",
)

# Map known Replicate model prefixes to their modalities.
_MODEL_MODALITY_HINTS: dict[str, Modality] = {
    "meta/": Modality.TEXT,
    "mistralai/": Modality.TEXT,
    "black-forest-labs/": Modality.IMAGE,
    "stability-ai/": Modality.IMAGE,
    "kwaivgi/": Modality.VIDEO,
    "resemble-ai/": Modality.AUDIO,
    "suno-ai/": Modality.AUDIO,
}


class ReplicateProvider(ModelProvider):
    """Replicate API provider supporting text, image, video, and audio models."""

    def __init__(self, api_token: str = "") -> None:
        self._api_token = api_token
        self._client: Any = None

    async def initialize(self) -> None:
        try:
            import replicate
        except ImportError as e:
            msg = "Install replicate: pip install 'mmsafe-bench[providers]'"
            raise ImportError(msg) from e

        if not self._api_token.strip():
            msg = "Replicate API token is missing (set REPLICATE_API_TOKEN)"
            raise RuntimeError(msg)

        self._client = replicate.Client(api_token=self._api_token)
        log.info("Replicate provider initialized")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._client is None:
            msg = "Provider not initialized. Call initialize() first."
            raise RuntimeError(msg)

        start = time.monotonic()

        try:
            # Build input based on modality
            input_params = self._build_input(request)

            output = self._client.run(request.model, input=input_params)
            latency = (time.monotonic() - start) * 1000

            content = self._extract_content(output, request.modality)
            content_type = self._content_type(request.modality)

            # Check for refusal in text output
            refusal = False
            if isinstance(content, str):
                refusal = any(phrase in content.lower() for phrase in _REFUSAL_PHRASES)

            return GenerationResponse(
                status=ProviderStatus.REFUSED if refusal else ProviderStatus.OK,
                content=content,
                content_type=content_type,
                model=request.model,
                provider_name="replicate",
                modality=request.modality,
                refusal_detected=refusal,
                refusal_reason="Refusal detected in response" if refusal else "",
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            log.warning("Replicate request failed: %s", exc)

            status = ProviderStatus.ERROR
            if "rate" in str(exc).lower():
                status = ProviderStatus.RATE_LIMITED

            return GenerationResponse(
                status=status,
                content=None,
                content_type="text/plain",
                model=request.model,
                provider_name="replicate",
                modality=request.modality,
                latency_ms=latency,
                raw_response={"error": str(exc)},
            )

    @staticmethod
    def _build_input(request: GenerationRequest) -> dict[str, Any]:
        """Build model-specific input dictionary."""
        base: dict[str, Any] = {"prompt": request.prompt}
        base.update(request.parameters)

        if request.modality == Modality.TEXT:
            base.setdefault("max_tokens", 1024)
            base.setdefault("temperature", 0.7)
        elif request.modality == Modality.IMAGE:
            base.setdefault("num_outputs", 1)
        elif request.modality == Modality.AUDIO:
            base.setdefault("text_input", request.prompt)
            if "prompt" in base and "text_input" in base:
                del base["prompt"]

        return base

    @staticmethod
    def _extract_content(output: Any, modality: Modality) -> str | bytes | None:
        """Extract content from Replicate output."""
        if output is None:
            return None

        # Replicate returns various types depending on model
        if isinstance(output, str):
            return output

        if isinstance(output, list):
            # Text models often return list of strings
            if all(isinstance(item, str) for item in output):
                return "".join(output)
            # Image/video models return list of URLs
            if output:
                return str(output[0])

        # Iterator for streaming text
        if hasattr(output, "__iter__"):
            parts = list(output)
            return "".join(str(p) for p in parts)

        return str(output)

    @staticmethod
    def _content_type(modality: Modality) -> str:
        return {
            Modality.TEXT: "text/plain",
            Modality.IMAGE: "image/png",
            Modality.VIDEO: "video/mp4",
            Modality.AUDIO: "audio/wav",
        }.get(modality, "text/plain")

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="replicate",
            supported_modalities=(Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
            supported_models=(
                "meta/meta-llama-3.1-405b-instruct",
                "black-forest-labs/flux-1.1-pro",
                "kwaivgi/kling-v2.6",
                "resemble-ai/chatterbox",
                "suno-ai/bark",
            ),
            max_concurrent_requests=5,
            supports_streaming=False,
            supports_multi_turn=False,
            rate_limit_rpm=60,
        )

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.get("meta/meta-llama-3.1-405b-instruct")
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        self._client = None
