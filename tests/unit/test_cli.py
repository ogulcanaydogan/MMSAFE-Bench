"""Tests for mmsafe.cli module using Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mmsafe.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestVersion:
    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "mmsafe" in result.output.lower()


class TestTaxonomyCommand:
    def test_taxonomy_lists_hazards(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["taxonomy"])
        assert result.exit_code == 0
        assert "S1" in result.output  # VIOLENT_CRIME

    def test_taxonomy_filter_text(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["taxonomy", "--modality", "text"])
        assert result.exit_code == 0
        assert "text" in result.output.lower() or "S1" in result.output


class TestProvidersCommand:
    def test_providers_lists_available(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["providers"])
        assert result.exit_code == 0
        assert "stub" in result.output.lower() or "openai" in result.output.lower()


class TestAttacksCommand:
    def test_attacks_lists_strategies(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["attacks"])
        assert result.exit_code == 0
        assert "passthrough" in result.output.lower()


class TestValidateCommand:
    def test_validate_valid_dataset(self, runner: CliRunner, tmp_path: Path) -> None:
        dataset = tmp_path / "valid.jsonl"
        dataset.write_text(
            json.dumps({
                "id": "t-001",
                "prompt": "Hello world",
                "category": "S1",
                "modality": "text",
            })
            + "\n"
        )
        result = runner.invoke(main, ["validate", "--dataset", str(dataset)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_nonexistent_file(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["validate", "--dataset", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0


class TestRunCommand:
    def test_dry_run_with_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Dry run should validate config and exit without running evaluation."""
        import yaml

        config_data = {
            "name": "test-eval",
            "version": "1.0",
            "models": [
                {"provider": "stub", "model": "stub-text-v1", "modalities": ["text"]},
            ],
            "datasets": [
                {"path": "datasets/text/mlcommons_hazards.jsonl", "modality": "text"},
            ],
            "attacks": [{"name": "passthrough"}],
            "judges": [{"name": "keyword_judge", "weight": 1.0}],
            "execution": {
                "concurrency": 2,
                "timeout_seconds": 30,
                "retry_attempts": 1,
                "checkpoint_interval": 10,
            },
            "metrics": {
                "confidence_interval": 0.95,
                "bootstrap_samples": 100,
            },
            "output": {
                "directory": str(tmp_path / "output"),
                "formats": ["json"],
            },
        }
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config_data, default_flow_style=False))

        result = runner.invoke(main, ["run", "--config", str(config_path), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
