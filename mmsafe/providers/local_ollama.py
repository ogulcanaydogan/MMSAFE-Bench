"""Local Ollama provider via HTTP API."""

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

log = get_logger("providers.local_ollama")

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


class OllamaProvider(ModelProvider):
    """Provider for a locally hosted Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=300.0)
        if not await self.health_check():
            await self._client.aclose()
            self._client = None
            msg = f"Ollama endpoint is not reachable at {self._base_url}"
            raise RuntimeError(msg)
        log.info("local_ollama provider initialized (base_url=%s)", self._base_url)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.modality != Modality.TEXT:
            return GenerationResponse(
                status=ProviderStatus.ERROR,
                content=None,
                content_type="text/plain",
                model=request.model,
                provider_name="local_ollama",
                modality=request.modality,
                raw_response={"error": f"Unsupported modality: {request.modality.value}"},
            )

        if self._client is None:
            msg = "Provider not initialized. Call initialize() first."
            raise RuntimeError(msg)

        start = time.monotonic()
        endpoint = f"{self._base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": self._build_prompt(request),
            "stream": False,
        }
        payload.update(request.parameters)

        try:
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            latency = (time.monotonic() - start) * 1000

            data: dict[str, Any] = response.json()
            content = data.get("response", "")
            content_str = content if isinstance(content, str) else str(content)
            refusal = any(phrase in content_str.lower() for phrase in _REFUSAL_PHRASES)

            usage: dict[str, int] = {}
            prompt_eval_count = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")
            if isinstance(prompt_eval_count, int):
                usage["prompt_tokens"] = prompt_eval_count
            if isinstance(eval_count, int):
                usage["completion_tokens"] = eval_count
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                    "completion_tokens", 0
                )

            return GenerationResponse(
                status=ProviderStatus.REFUSED if refusal else ProviderStatus.OK,
                content=content_str,
                content_type="text/plain",
                model=request.model,
                provider_name="local_ollama",
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
            name="local_ollama",
            supported_modalities=(Modality.TEXT,),
            supported_models=("llama3.1:8b", "mistral:7b", "any-ollama-model"),
            max_concurrent_requests=4,
            supports_streaming=False,
            supports_multi_turn=True,
            rate_limit_rpm=None,
        )

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            return response.status_code < 400
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _build_prompt(request: GenerationRequest) -> str:
        if not request.conversation_history:
            return request.prompt

        lines: list[str] = []
        for msg in request.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append(f"user: {request.prompt}")
        return "\n".join(lines)

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
            provider_name="local_ollama",
            modality=request.modality,
            latency_ms=(time.monotonic() - start) * 1000,
            raw_response={"error": error_text},
        )
