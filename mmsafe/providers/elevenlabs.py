"""ElevenLabs provider for audio generation and voice cloning."""

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

log = get_logger("providers.elevenlabs")


class ElevenLabsProvider(ModelProvider):
    """ElevenLabs API provider for text-to-speech and voice cloning."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._client: Any = None

    async def initialize(self) -> None:
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "Install elevenlabs: pip install 'mmsafe-bench[providers]'"
            raise ImportError(msg) from e

        self._client = ElevenLabs(api_key=self._api_key)
        log.info("ElevenLabs provider initialized")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._client is None:
            msg = "Provider not initialized. Call initialize() first."
            raise RuntimeError(msg)

        start = time.monotonic()

        try:
            voice_id = request.parameters.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
            model_id = request.model or "eleven_multilingual_v2"
            stability = request.parameters.get("stability", 0.5)
            similarity_boost = request.parameters.get("similarity_boost", 0.75)

            audio = self._client.text_to_speech.convert(
                text=request.prompt,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings={
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
            )

            # Collect audio bytes from generator
            audio_bytes = b""
            if hasattr(audio, "__iter__"):
                for chunk in audio:
                    if isinstance(chunk, bytes):
                        audio_bytes += chunk
            elif isinstance(audio, bytes):
                audio_bytes = audio

            latency = (time.monotonic() - start) * 1000

            if not audio_bytes:
                return GenerationResponse(
                    status=ProviderStatus.REFUSED,
                    content=None,
                    content_type="audio/mpeg",
                    model=model_id,
                    provider_name="elevenlabs",
                    modality=Modality.AUDIO,
                    refusal_detected=True,
                    refusal_reason="Audio generation returned empty content",
                    latency_ms=latency,
                )

            return GenerationResponse(
                status=ProviderStatus.OK,
                content=audio_bytes,
                content_type="audio/mpeg",
                model=model_id,
                provider_name="elevenlabs",
                modality=Modality.AUDIO,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            log.warning("ElevenLabs request failed: %s", exc)

            status = ProviderStatus.ERROR
            err_str = str(exc).lower()
            if "rate" in err_str or "quota" in err_str:
                status = ProviderStatus.RATE_LIMITED
            elif "content" in err_str and ("policy" in err_str or "safety" in err_str):
                status = ProviderStatus.REFUSED

            return GenerationResponse(
                status=status,
                content=None,
                content_type="audio/mpeg",
                model=request.model,
                provider_name="elevenlabs",
                modality=Modality.AUDIO,
                latency_ms=latency,
                raw_response={"error": str(exc)},
            )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="elevenlabs",
            supported_modalities=(Modality.AUDIO,),
            supported_models=(
                "eleven_multilingual_v2",
                "eleven_turbo_v2.5",
                "eleven_english_sts_v2",
            ),
            max_concurrent_requests=5,
            supports_streaming=True,
            supports_multi_turn=False,
            rate_limit_rpm=100,
        )

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.voices.get_all()
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        self._client = None
