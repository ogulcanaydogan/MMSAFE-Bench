"""Tests for the evaluation pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mmsafe.attacks.base import AttackMetadata
from mmsafe.judges.base import SafetyVerdict
from mmsafe.pipeline.checkpointing import load_checkpoint, save_checkpoint
from mmsafe.pipeline.result_types import EvalRun, EvalSample
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import HazardCategory, Modality

if TYPE_CHECKING:
    from pathlib import Path


def _make_sample(sample_id: str = "test-001") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        prompt_id="p-001",
        modality=Modality.TEXT,
        hazard_category=HazardCategory.VIOLENT_CRIME,
        original_prompt="test",
        attack_metadata=AttackMetadata(strategy_name="passthrough", original_prompt="test"),
        request=GenerationRequest(prompt="test", modality=Modality.TEXT, model="stub"),
        response=GenerationResponse(
            status=ProviderStatus.OK,
            content="response",
            content_type="text/plain",
            model="stub",
            provider_name="stub",
            modality=Modality.TEXT,
        ),
        verdict=SafetyVerdict(is_safe=True, confidence=0.9, judge_name="test"),
        attack_succeeded=False,
        model_refused=False,
    )


class TestEvalRun:
    def test_creation(self) -> None:
        run = EvalRun(run_id="test-run", config_name="test")
        assert run.run_id == "test-run"
        assert len(run.samples) == 0

    def test_to_dict(self) -> None:
        run = EvalRun(run_id="test-run", config_name="test")
        run.samples.append(_make_sample())
        run.completed_at = datetime.now(UTC)

        data = run.to_dict()
        assert data["run_id"] == "test-run"
        assert data["total_samples"] == 1
        assert len(data["samples"]) == 1
        assert data["samples"][0]["sample_id"] == "test-001"

    def test_duration(self) -> None:
        run = EvalRun(
            run_id="test",
            config_name="test",
            started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC),
        )
        assert run.duration_seconds == 60.0


class TestCheckpointing:
    def test_save_and_load(self, tmp_path: Path) -> None:
        run = EvalRun(run_id="ckpt-test", config_name="test")
        run.samples.append(_make_sample("s1"))
        run.samples.append(_make_sample("s2"))

        save_checkpoint(run, tmp_path)

        completed = load_checkpoint(tmp_path / "checkpoint_ckpt-test.json")
        assert completed == {"s1", "s2"}

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        completed = load_checkpoint(tmp_path / "nonexistent.json")
        assert completed == set()

    def test_save_merges_base_completed_ids(self, tmp_path: Path) -> None:
        run = EvalRun(run_id="ckpt-merge", config_name="test")
        run.samples.append(_make_sample("s2"))
        run.samples.append(_make_sample("s3"))

        path = save_checkpoint(run, tmp_path, base_completed_ids={"s1", "s2"})
        data = json.loads(path.read_text())

        assert data["completed_sample_ids"] == ["s1", "s2", "s3"]
        assert data["completed_count"] == 3
        assert data["total_samples"] == 2

        completed = load_checkpoint(path)
        assert completed == {"s1", "s2", "s3"}


class TestEvalSample:
    def test_frozen(self) -> None:
        sample = _make_sample()
        with pytest.raises(AttributeError):
            sample.sample_id = "changed"  # type: ignore[misc]
