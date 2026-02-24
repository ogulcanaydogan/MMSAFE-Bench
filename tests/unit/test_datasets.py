"""Tests for dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmsafe.datasets.loader import load_dataset, validate_dataset


class TestValidateDataset:
    def test_valid_dataset(self, tmp_path: Path) -> None:
        ds = tmp_path / "valid.jsonl"
        ds.write_text(
            '{"id": "001", "prompt": "test prompt", "category": "S1"}\n'
            '{"id": "002", "prompt": "another prompt", "category": "S10"}\n'
        )
        errors = validate_dataset(ds)
        assert errors == []

    def test_missing_fields(self, tmp_path: Path) -> None:
        ds = tmp_path / "missing.jsonl"
        ds.write_text('{"id": "001", "prompt": "test"}\n')
        errors = validate_dataset(ds)
        assert len(errors) == 1
        assert "category" in errors[0]

    def test_invalid_category(self, tmp_path: Path) -> None:
        ds = tmp_path / "bad_cat.jsonl"
        ds.write_text('{"id": "001", "prompt": "test", "category": "INVALID"}\n')
        errors = validate_dataset(ds)
        assert len(errors) == 1
        assert "Invalid category" in errors[0]

    def test_duplicate_ids(self, tmp_path: Path) -> None:
        ds = tmp_path / "dupes.jsonl"
        ds.write_text(
            '{"id": "001", "prompt": "test", "category": "S1"}\n'
            '{"id": "001", "prompt": "dupe", "category": "S1"}\n'
        )
        errors = validate_dataset(ds)
        assert any("Duplicate" in e for e in errors)

    def test_empty_prompt(self, tmp_path: Path) -> None:
        ds = tmp_path / "empty.jsonl"
        ds.write_text('{"id": "001", "prompt": "", "category": "S1"}\n')
        errors = validate_dataset(ds)
        assert any("Empty prompt" in e for e in errors)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        errors = validate_dataset(tmp_path / "nope.jsonl")
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_invalid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "bad.jsonl"
        ds.write_text("not json at all\n")
        errors = validate_dataset(ds)
        assert any("Invalid JSON" in e for e in errors)

    def test_custom_categories(self, tmp_path: Path) -> None:
        ds = tmp_path / "custom.jsonl"
        ds.write_text('{"id": "001", "prompt": "test", "category": "X1"}\n')
        errors = validate_dataset(ds)
        assert errors == []

    def test_valid_modality(self, tmp_path: Path) -> None:
        ds = tmp_path / "mod.jsonl"
        ds.write_text('{"id": "001", "prompt": "test", "category": "S1", "modality": "image"}\n')
        errors = validate_dataset(ds)
        assert errors == []

    def test_invalid_modality(self, tmp_path: Path) -> None:
        ds = tmp_path / "badmod.jsonl"
        ds.write_text('{"id": "001", "prompt": "test", "category": "S1", "modality": "smell"}\n')
        errors = validate_dataset(ds)
        assert any("Invalid modality" in e for e in errors)


class TestLoadDataset:
    def test_load(self, tmp_path: Path) -> None:
        ds = tmp_path / "test.jsonl"
        ds.write_text(
            '{"id": "001", "prompt": "test", "category": "S1"}\n'
            '{"id": "002", "prompt": "test2", "category": "S2"}\n'
        )
        entries = load_dataset(ds)
        assert len(entries) == 2
        assert entries[0]["id"] == "001"

    def test_max_samples(self, tmp_path: Path) -> None:
        lines = [f'{{"id": "{i:03}", "prompt": "test", "category": "S1"}}\n' for i in range(20)]
        ds = tmp_path / "many.jsonl"
        ds.write_text("".join(lines))
        entries = load_dataset(ds, max_samples=5)
        assert len(entries) == 5


class TestSeedDatasets:
    """Validate that all seed datasets in the repository are valid."""

    def test_all_seed_datasets_valid(self) -> None:
        datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        for jsonl_file in datasets_dir.rglob("*.jsonl"):
            errors = validate_dataset(jsonl_file)
            assert errors == [], f"{jsonl_file.name}: {errors}"
