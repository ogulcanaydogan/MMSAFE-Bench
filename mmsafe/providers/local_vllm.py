"""Local vLLM provider using the OpenAI-compatible HTTP API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from mmsafe._internal.logging import get_logger
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderCapabilities,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import Modality

log = get_logger("providers.local_vllm")

_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "cannot assist",
    "content policy",
    "not allowed",
    "i must decline",
)


class VLLMProvider(ModelProvider):
    """Provider for a locally hosted vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=120.0)
        if not await self.health_check():
            await self._client.aclose()
            self._client = None
            msg = f"vLLM endpoint is not reachable at {self._base_url}"
            raise RuntimeError(msg)
        log.info("local_vllm provider initialized (base_url=%s)", self._base_url)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.modality != Modality.TEXT:
            return GenerationResponse(
                status=ProviderStatus.ERROR,
                content=None,
                content_type="text/plain",
                model=request.model,
                provider_name="local_vllm",
                modality=request.modality,
                raw_response={"error": f"Unsupported modality: {request.modality.value}"},
            )

        if self._client is None:
            msg = "Provider not initialized. Call initialize() first."
            raise RuntimeError(msg)

        start = time.monotonic()
        endpoint = f"{self._base_url}/v1/chat/completions"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        messages = [*request.conversation_history, {"role": "user", "content": request.prompt}]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        payload.update(request.parameters)
        payload.setdefault("temperature", 0.7)
        payload.setdefault("max_tokens", 1024)

        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            latency = (time.monotonic() - start) * 1000

            data: dict[str, Any] = response.json()
            content = self._extract_text(data)
            refusal = any(phrase in content.lower() for phrase in _REFUSAL_PHRASES)

            usage_raw = data.get("usage", {})
            usage: dict[str, int] = {}
            if isinstance(usage_raw, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage_raw.get(key)
                    if isinstance(value, int):
                        usage[key] = value

            return GenerationResponse(
                status=ProviderStatus.REFUSED if refusal else ProviderStatus.OK,
                content=content,
                content_type="text/plain",
                model=request.model,
                provider_name="local_vllm",
                modality=Modality.TEXT,
                refusal_detected=refusal,
                refusal_reason="Refusal language detected" if refusal else "",
                latency_ms=latency,
                raw_response=data,
                usage=usage,
            )
        except httpx.ReadTimeout as exc:
            return self._failure_response(request, start, ProviderStatus.TIMEOUT, str(exc))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                status = ProviderStatus.RATE_LIMITED
            else:
                status = ProviderStatus.ERROR
            return self._failure_response(request, start, status, str(exc))
        except Exception as exc:
            return self._failure_response(request, start, ProviderStatus.ERROR, str(exc))

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="local_vllm",
            supported_modalities=(Modality.TEXT,),
            supported_models=("any-hf-model",),
            max_concurrent_requests=16,
            supports_streaming=False,
            supports_multi_turn=True,
            rate_limit_rpm=None,
        )

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            health = await self._client.get(f"{self._base_url}/health")
            if health.status_code < 400:
                return True
            models = await self._client.get(f"{self._base_url}/v1/models")
            return models.status_code < 400
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
        return ""

    @staticmethod
    def _failure_response(
        request: GenerationRequest,
        start: float,
        status: ProviderStatus,
        error_text: str,
    ) -> GenerationResponse:
        return GenerationResponse(
            status=status,
            content=None,
            content_type="text/plain",
            model=request.model,
            provider_name="local_vllm",
            modality=request.modality,
            latency_ms=(time.monotonic() - start) * 1000,
            raw_response={"error": error_text},
        )
