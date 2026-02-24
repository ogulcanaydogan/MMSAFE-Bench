"""Tests for mmsafe.pipeline.executor module."""

from __future__ import annotations

import asyncio

import pytest

from mmsafe.judges.base import SafetyJudge, SafetyVerdict
from mmsafe.pipeline.executor import PipelineExecutor
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderCapabilities,
    ProviderStatus,
)
from mmsafe.providers.stub import StubProvider
from mmsafe.taxonomy.categories import Modality


# --- Fixtures ----------------------------------------------------------

def _text_request(prompt: str = "Hello world") -> GenerationRequest:
    return GenerationRequest(prompt=prompt, modality=Modality.TEXT, model="stub-text-v1")


class _AlwaysSafeJudge(SafetyJudge):
    """Minimal judge stub that always returns safe."""

    @property
    def name(self) -> str:
        return "always_safe"

    @property
    def supported_modalities(self) -> tuple[Modality, ...]:
        return (Modality.TEXT,)

    async def initialize(self) -> None:
        pass

    async def evaluate(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> SafetyVerdict:
        return SafetyVerdict(
            is_safe=True,
            confidence=1.0,
            explanation="Always safe",
            judge_name=self.name,
        )

    async def shutdown(self) -> None:
        pass


class _SlowProvider(ModelProvider):
    """Provider that takes too long to respond."""

    def __init__(self, delay: float = 5.0) -> None:
        self._delay = delay

    async def initialize(self) -> None:
        pass

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        await asyncio.sleep(self._delay)
        return GenerationResponse(
            status=ProviderStatus.OK,
            content="slow response",
            content_type="text/plain",
            model=request.model,
            provider_name="slow",
            modality=request.modality,
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="slow",
            supported_modalities=(Modality.TEXT,),
            supported_models=("slow-v1",),
        )

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        pass


# --- Tests -------------------------------------------------------------

class TestExecutorGenerate:
    @pytest.mark.asyncio
    async def test_successful_generation(self) -> None:
        stub = StubProvider()
        await stub.initialize()
        executor = PipelineExecutor(
            providers={"stub": stub},
            judges=[_AlwaysSafeJudge()],
            max_concurrency=2,
            timeout_seconds=10,
            retry_attempts=1,
        )
        resp = await executor.generate("stub", _text_request())
        assert resp.status == ProviderStatus.OK
        assert resp.content is not None
        await stub.shutdown()

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error(self) -> None:
        executor = PipelineExecutor(
            providers={},
            judges=[_AlwaysSafeJudge()],
        )
        resp = await executor.generate("nonexistent", _text_request())
        assert resp.status == ProviderStatus.ERROR
        assert "not found" in str(resp.raw_response.get("error", "")).lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_status(self) -> None:
        slow = _SlowProvider(delay=5.0)
        executor = PipelineExecutor(
            providers={"slow": slow},
            judges=[],
            max_concurrency=1,
            timeout_seconds=1,
            retry_attempts=1,
        )
        resp = await executor.generate("slow", _text_request())
        assert resp.status in (ProviderStatus.TIMEOUT, ProviderStatus.ERROR)


class TestExecutorJudge:
    @pytest.mark.asyncio
    async def test_returns_verdict(self) -> None:
        judge = _AlwaysSafeJudge()
        executor = PipelineExecutor(providers={}, judges=[judge])
        resp = GenerationResponse(
            status=ProviderStatus.OK,
            content="hello",
            content_type="text/plain",
            model="m",
            provider_name="p",
            modality=Modality.TEXT,
        )
        verdict = await executor.judge(_text_request(), resp)
        assert verdict.is_safe is True
        assert verdict.judge_name == "always_safe"

    @pytest.mark.asyncio
    async def test_no_judges_returns_default_safe(self) -> None:
        executor = PipelineExecutor(providers={}, judges=[])
        resp = GenerationResponse(
            status=ProviderStatus.OK,
            content="hello",
            content_type="text/plain",
            model="m",
            provider_name="p",
            modality=Modality.TEXT,
        )
        verdict = await executor.judge(_text_request(), resp)
        assert verdict.is_safe is True
        assert verdict.confidence == 0.0
