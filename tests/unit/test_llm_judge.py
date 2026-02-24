"""Tests for mmsafe.judges.llm_judge module (mocked, no API calls)."""

from __future__ import annotations

import json

import pytest

from mmsafe.judges.llm_judge import LLMJudge
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import HazardCategory, Modality


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="test prompt", modality=Modality.TEXT, model="m")


def _response() -> GenerationResponse:
    return GenerationResponse(
        status=ProviderStatus.OK,
        content="response text",
        content_type="text/plain",
        model="m",
        provider_name="p",
        modality=Modality.TEXT,
    )


class TestLLMJudgeProperties:
    def test_name(self) -> None:
        judge = LLMJudge()
        assert judge.name == "llm_judge"

    def test_supported_modalities(self) -> None:
        judge = LLMJudge()
        mods = judge.supported_modalities
        assert Modality.TEXT in mods
        assert Modality.IMAGE in mods


class TestParseVerdict:
    def test_valid_json_safe(self) -> None:
        judge = LLMJudge()
        result_text = json.dumps({
            "is_safe": True,
            "confidence": 0.95,
            "hazard_categories": [],
            "explanation": "No issues found",
        })
        verdict = judge._parse_verdict(result_text, latency_ms=10.0)
        assert verdict.is_safe is True
        assert verdict.confidence == 0.95
        assert verdict.explanation == "No issues found"
        assert verdict.latency_ms == 10.0

    def test_valid_json_unsafe(self) -> None:
        judge = LLMJudge()
        result_text = json.dumps({
            "is_safe": False,
            "confidence": 0.8,
            "hazard_categories": ["VIOLENT_CRIME", "HATE_SPEECH"],
            "explanation": "Contains harmful content",
        })
        verdict = judge._parse_verdict(result_text, latency_ms=5.0)
        assert verdict.is_safe is False
        assert verdict.confidence == 0.8
        assert HazardCategory.VIOLENT_CRIME in verdict.hazard_categories
        assert HazardCategory.HATE_SPEECH in verdict.hazard_categories

    def test_malformed_json_returns_safe_default(self) -> None:
        judge = LLMJudge()
        verdict = judge._parse_verdict("not json at all", latency_ms=1.0)
        assert verdict.is_safe is True
        assert verdict.confidence == 0.0
        assert "Failed to parse" in verdict.explanation

    def test_unknown_hazard_category_ignored(self) -> None:
        judge = LLMJudge()
        result_text = json.dumps({
            "is_safe": False,
            "confidence": 0.7,
            "hazard_categories": ["VIOLENT_CRIME", "MADE_UP_CATEGORY"],
            "explanation": "test",
        })
        verdict = judge._parse_verdict(result_text, latency_ms=1.0)
        assert len(verdict.hazard_categories) == 1
        assert HazardCategory.VIOLENT_CRIME in verdict.hazard_categories


class TestEvaluateWithoutInit:
    @pytest.mark.asyncio
    async def test_returns_safe_with_zero_confidence(self) -> None:
        """An uninitialized judge (no client) should return safe with 0 confidence."""
        judge = LLMJudge()
        verdict = await judge.evaluate(_request(), _response())
        assert verdict.is_safe is True
        assert verdict.confidence == 0.0
        assert "not initialized" in verdict.explanation.lower()
