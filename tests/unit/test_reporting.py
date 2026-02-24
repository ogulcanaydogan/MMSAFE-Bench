"""Tests for the reporting system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mmsafe.attacks.base import AttackMetadata
from mmsafe.judges.base import SafetyVerdict
from mmsafe.pipeline.result_types import EvalRun, EvalSample
from mmsafe.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderStatus,
)
from mmsafe.taxonomy.categories import HazardCategory, Modality


def _sample(
    *,
    sample_id: str = "s1",
    model: str = "gpt-4o",
    provider: str = "openai",
    attack_succeeded: bool = False,
    model_refused: bool = False,
    category: HazardCategory = HazardCategory.VIOLENT_CRIME,
    attack_name: str = "passthrough",
) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        prompt_id=f"p-{sample_id}",
        modality=Modality.TEXT,
        hazard_category=category,
        original_prompt="test prompt",
        attack_metadata=AttackMetadata(strategy_name=attack_name, original_prompt="test"),
        request=GenerationRequest(prompt="test", modality=Modality.TEXT, model=model),
        response=GenerationResponse(
            status=ProviderStatus.OK,
            content="response",
            content_type="text/plain",
            model=model,
            provider_name=provider,
            modality=Modality.TEXT,
            refusal_detected=model_refused,
        ),
        verdict=SafetyVerdict(is_safe=not attack_succeeded, confidence=0.9, judge_name="test"),
        attack_succeeded=attack_succeeded,
        model_refused=model_refused,
    )


def _make_run() -> EvalRun:
    run = EvalRun(
        run_id="test-run-001",
        config_name="test-eval",
        started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc),
    )
    run.samples = [
        _sample(sample_id="s1", model="gpt-4o", attack_succeeded=True, attack_name="jailbreak"),
        _sample(sample_id="s2", model="gpt-4o", attack_succeeded=False, attack_name="passthrough"),
        _sample(
            sample_id="s3",
            model="claude-3",
            provider="anthropic",
            attack_succeeded=False,
            category=HazardCategory.HATE_SPEECH,
        ),
        _sample(
            sample_id="s4",
            model="claude-3",
            provider="anthropic",
            attack_succeeded=True,
            model_refused=False,
            category=HazardCategory.HATE_SPEECH,
            attack_name="jailbreak",
        ),
    ]
    return run


class TestJSONReporter:
    def test_generate_json_report(self, tmp_path: Path) -> None:
        from mmsafe.reporting.json_reporter import generate_json_report

        run = _make_run()
        out = tmp_path / "report.json"
        result_path = generate_json_report(run, out)

        assert result_path == out
        assert out.exists()

        data = json.loads(out.read_text())
        assert data["framework"] == "mmsafe-bench"
        assert data["run_id"] == "test-run-001"
        assert "summary" in data
        assert data["summary"]["overall"]["total_samples"] == 4

    def test_load_json_report(self, tmp_path: Path) -> None:
        from mmsafe.reporting.json_reporter import generate_json_report, load_json_report

        run = _make_run()
        out = tmp_path / "report.json"
        generate_json_report(run, out)

        loaded = load_json_report(out)
        assert loaded["run_id"] == "test-run-001"


class TestMarkdownReporter:
    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        from mmsafe.reporting.markdown_reporter import generate_markdown_report

        run = _make_run()
        out = tmp_path / "report.md"
        result_path = generate_markdown_report(run, out)

        assert result_path == out
        assert out.exists()

        content = out.read_text()
        assert "# MMSAFE-Bench Safety Report" in content
        assert "test-run-001" in content
        assert "Attack Success Rate" in content
        assert "Results by Model" in content
        assert "Results by Attack Strategy" in content

    def test_empty_run(self, tmp_path: Path) -> None:
        from mmsafe.reporting.markdown_reporter import generate_markdown_report

        run = EvalRun(run_id="empty", config_name="empty-test")
        out = tmp_path / "empty.md"
        generate_markdown_report(run, out)

        content = out.read_text()
        assert "MMSAFE-Bench" in content


class TestHTMLReporter:
    def test_generate_html_report(self, tmp_path: Path) -> None:
        from mmsafe.reporting.html_reporter import generate_html_report

        run = _make_run()
        out = tmp_path / "report.html"
        result_path = generate_html_report(run, out)

        assert result_path == out
        assert out.exists()

        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "MMSAFE-Bench" in content
        assert "test-eval" in content
        assert "plotly" in content.lower()


class TestCharts:
    def test_asr_by_category_chart(self) -> None:
        from mmsafe.reporting.charts import create_asr_by_category_chart

        data = {"S1": {"asr": 0.3}, "S10": {"asr": 0.8}}
        html = create_asr_by_category_chart(data)
        assert "plotly" in html.lower() or "div" in html.lower()

    def test_asr_by_model_chart(self) -> None:
        from mmsafe.reporting.charts import create_asr_by_model_chart

        data = {"gpt-4o": {"asr": 0.2}, "claude-3": {"asr": 0.1}}
        html = create_asr_by_model_chart(data)
        assert len(html) > 0

    def test_radar_chart(self) -> None:
        from mmsafe.reporting.charts import create_radar_chart

        data = {"S1": {"asr": 0.3}, "S10": {"asr": 0.5}, "S2": {"asr": 0.1}}
        html = create_radar_chart(data)
        assert len(html) > 0

    def test_refusal_heatmap(self) -> None:
        from mmsafe.reporting.charts import create_refusal_heatmap

        data = {"S1": {"rr": 0.5}, "S10": {"rr": 0.8}}
        html = create_refusal_heatmap(data)
        assert len(html) > 0


class TestLeaderboard:
    def test_build_leaderboard(self) -> None:
        from mmsafe.reporting.leaderboard import build_leaderboard

        run = _make_run()
        entries = build_leaderboard(run)

        assert len(entries) == 2
        # Entries sorted by safety score descending
        assert entries[0].safety_score >= entries[1].safety_score

    def test_save_leaderboard_json(self, tmp_path: Path) -> None:
        from mmsafe.reporting.leaderboard import (
            LeaderboardEntry,
            save_leaderboard_json,
        )

        entries = [
            LeaderboardEntry(model="model-a", safety_score=90.0, asr=0.1, rr=0.3, samples=100),
            LeaderboardEntry(model="model-b", safety_score=70.0, asr=0.3, rr=0.2, samples=100),
        ]
        out = tmp_path / "lb.json"
        save_leaderboard_json(entries, out)

        data = json.loads(out.read_text())
        assert len(data) == 2
        assert data[0]["rank"] == 1
        assert data[0]["model"] == "model-a"

    def test_merge_leaderboard(self, tmp_path: Path) -> None:
        from mmsafe.reporting.leaderboard import (
            LeaderboardEntry,
            merge_leaderboard,
            save_leaderboard_json,
        )

        existing = [
            LeaderboardEntry(model="model-a", safety_score=80.0, asr=0.2, rr=0.3, samples=50),
        ]
        lb_path = tmp_path / "lb.json"
        save_leaderboard_json(existing, lb_path)

        new_entries = [
            LeaderboardEntry(model="model-a", safety_score=90.0, asr=0.1, rr=0.3, samples=100),
            LeaderboardEntry(model="model-b", safety_score=70.0, asr=0.3, rr=0.2, samples=100),
        ]

        merged = merge_leaderboard(lb_path, new_entries)
        assert len(merged) == 2
        # model-a should have new higher score
        model_a = next(e for e in merged if e.model == "model-a")
        assert model_a.safety_score == 90.0

    def test_generate_leaderboard_html(self, tmp_path: Path) -> None:
        from mmsafe.reporting.leaderboard import (
            LeaderboardEntry,
            generate_leaderboard_html,
        )

        entries = [
            LeaderboardEntry(model="safe-model", safety_score=95.0, asr=0.05, rr=0.4, samples=200),
            LeaderboardEntry(model="risky-model", safety_score=40.0, asr=0.6, rr=0.1, samples=200),
        ]
        out = tmp_path / "leaderboard.html"
        generate_leaderboard_html(entries, out)

        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Safety Leaderboard" in content
        assert "safe-model" in content
