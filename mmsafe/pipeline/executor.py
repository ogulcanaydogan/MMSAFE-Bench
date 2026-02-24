"""Async execution engine with bounded concurrency."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from mmsafe._internal.logging import get_logger
from mmsafe._internal.retry import retry_with_backoff
from mmsafe.judges.base import SafetyJudge, SafetyVerdict
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderStatus,
)

log = get_logger("pipeline.executor")


class PipelineExecutor:
    """Async execution engine for the evaluation pipeline.

    Handles concurrent API calls with rate limiting and retries.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider],
        judges: Sequence[SafetyJudge],
        max_concurrency: int = 5,
        timeout_seconds: int = 120,
        retry_attempts: int = 3,
    ) -> None:
        self._providers = providers
        self._judges = judges
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = timeout_seconds
        self._retries = retry_attempts

    async def generate(
        self,
        provider_name: str,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Execute a generation request with concurrency control and retries."""
        provider = self._providers.get(provider_name)
        if provider is None:
            return GenerationResponse(
                status=ProviderStatus.ERROR,
                content=None,
                content_type="text/plain",
                model=request.model,
                provider_name=provider_name,
                modality=request.modality,
                raw_response={"error": f"Provider not found: {provider_name}"},
            )

        async with self._semaphore:
            try:
                response = await retry_with_backoff(
                    lambda: asyncio.wait_for(
                        provider.generate(request),
                        timeout=self._timeout,
                    ),
                    max_attempts=self._retries,
                    retryable_exceptions=(asyncio.TimeoutError, ConnectionError, OSError),
                )
                return response
            except asyncio.TimeoutError:
                return GenerationResponse(
                    status=ProviderStatus.TIMEOUT,
                    content=None,
                    content_type="text/plain",
                    model=request.model,
                    provider_name=provider_name,
                    modality=request.modality,
                )
            except Exception as exc:
                log.warning("Generation failed for %s: %s", provider_name, exc)
                return GenerationResponse(
                    status=ProviderStatus.ERROR,
                    content=None,
                    content_type="text/plain",
                    model=request.model,
                    provider_name=provider_name,
                    modality=request.modality,
                    raw_response={"error": str(exc)},
                )

    async def judge(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> SafetyVerdict:
        """Run all judges and return the primary verdict."""
        if not self._judges:
            return SafetyVerdict(
                is_safe=True,
                confidence=0.0,
                explanation="No judges configured",
                judge_name="none",
            )

        # Use the first judge as primary (or composite if configured)
        judge = self._judges[0]
        return await judge.evaluate(request, response)
