"""Tests for mmsafe.judges.human_eval module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmsafe.judges.human_eval import HumanEvalJudge
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import Modality


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="test prompt", modality=Modality.TEXT, model="m")


def _response(content: str = "response text") -> GenerationResponse:
    return GenerationResponse(
        status=ProviderStatus.OK,
        content=content,
        content_type="text/plain",
        model="m",
        provider_name="p",
        modality=Modality.TEXT,
    )


class TestHumanEvalJudge:
    @pytest.mark.asyncio
    async def test_evaluate_queues_sample(self, tmp_path: Path) -> None:
        judge = HumanEvalJudge(output_path=str(tmp_path / "queue.jsonl"))
        await judge.initialize()

        verdict = await judge.evaluate(_request(), _response())
        assert verdict.is_safe is True
        assert verdict.confidence == 0.0
        assert judge.pending_count == 1

    @pytest.mark.asyncio
    async def test_pending_count_increments(self, tmp_path: Path) -> None:
        judge = HumanEvalJudge(output_path=str(tmp_path / "queue.jsonl"))
        await judge.initialize()

        await judge.evaluate(_request(), _response("a"))
        await judge.evaluate(_request(), _response("b"))
        assert judge.pending_count == 2

    @pytest.mark.asyncio
    async def test_export_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "queue.jsonl"
        judge = HumanEvalJudge(output_path=str(out))
        await judge.initialize()

        await judge.evaluate(_request(), _response("hello"))
        path = judge.export()

        assert path == out
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["prompt"] == "test prompt"
        assert entry["response_content"] == "hello"
        assert entry["needs_human_review"] is True

    @pytest.mark.asyncio
    async def test_export_clears_pending(self, tmp_path: Path) -> None:
        judge = HumanEvalJudge(output_path=str(tmp_path / "queue.jsonl"))
        await judge.initialize()

        await judge.evaluate(_request(), _response())
        assert judge.pending_count == 1
        judge.export()
        assert judge.pending_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_auto_exports(self, tmp_path: Path) -> None:
        out = tmp_path / "queue.jsonl"
        judge = HumanEvalJudge(output_path=str(out))
        await judge.initialize()

        await judge.evaluate(_request(), _response())
        await judge.shutdown()

        assert out.exists()
        assert judge.pending_count == 0

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "queue.jsonl"
        judge = HumanEvalJudge(output_path=str(nested))
        await judge.initialize()
        assert nested.parent.exists()

    def test_name_property(self) -> None:
        judge = HumanEvalJudge()
        assert judge.name == "human_eval"

    def test_supported_modalities(self) -> None:
        judge = HumanEvalJudge()
        mods = judge.supported_modalities
        assert Modality.TEXT in mods
        assert Modality.IMAGE in mods
