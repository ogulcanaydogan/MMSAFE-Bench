"""Tests for evaluation runner behavior."""

from __future__ import annotations

import pytest

from mmsafe.config.models import DatasetSpec, EvalConfig, JudgeSpec, ModelSpec
from mmsafe.pipeline.runner import EvalRunner


def _base_config() -> EvalConfig:
    return EvalConfig(
        models=[ModelSpec(provider="stub", model="stub-text-v1", modalities=["text"])],
        datasets=[DatasetSpec(path="datasets/text/mlcommons_hazards.jsonl", modality="text")],
        judges=[JudgeSpec(name="keyword_judge")],
    )


class TestProviderInitHandling:
    @pytest.mark.asyncio
    async def test_non_strict_skips_provider_init_errors(self, monkeypatch) -> None:
        runner = EvalRunner(_base_config())

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr("mmsafe.pipeline.runner.get_provider_class", _raise)
        providers = await runner._build_providers(strict=False)

        assert providers == {}
        assert "stub" in runner._provider_init_errors

    @pytest.mark.asyncio
    async def test_strict_raises_on_provider_init_errors(self, monkeypatch) -> None:
        runner = EvalRunner(_base_config())

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr("mmsafe.pipeline.runner.get_provider_class", _raise)

        with pytest.raises(RuntimeError, match="Failed to resolve provider"):
            await runner._build_providers(strict=True)
