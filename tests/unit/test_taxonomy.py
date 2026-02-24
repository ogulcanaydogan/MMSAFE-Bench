"""Tests for the safety hazard taxonomy module."""

from __future__ import annotations

from mmsafe.taxonomy.categories import (
    ALL_MODALITIES,
    HAZARD_TAXONOMY,
    HazardCategory,
    HazardSpec,
    Modality,
    Severity,
    get_custom_hazards,
    get_hazards_for_modality,
    get_mlcommons_hazards,
)


class TestModality:
    def test_all_modalities_defined(self) -> None:
        assert len(Modality) == 4
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.VIDEO.value == "video"
        assert Modality.AUDIO.value == "audio"

    def test_all_modalities_tuple(self) -> None:
        assert len(ALL_MODALITIES) == 4
        assert set(ALL_MODALITIES) == set(Modality)


class TestHazardCategory:
    def test_mlcommons_categories_s1_to_s12(self) -> None:
        mlcommons_codes = [f"S{i}" for i in range(1, 13)]
        for code in mlcommons_codes:
            assert HazardCategory(code) is not None

    def test_custom_categories_x1_to_x8(self) -> None:
        custom_codes = [f"X{i}" for i in range(1, 9)]
        for code in custom_codes:
            assert HazardCategory(code) is not None

    def test_total_categories(self) -> None:
        assert len(HazardCategory) == 20  # 12 MLCommons + 8 custom


class TestSeverity:
    def test_severity_ordering(self) -> None:
        assert Severity.LOW < Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL < Severity.CATASTROPHIC

    def test_severity_values(self) -> None:
        assert Severity.LOW.value == 1
        assert Severity.CATASTROPHIC.value == 5


class TestHazardTaxonomy:
    def test_all_categories_have_specs(self) -> None:
        for category in HazardCategory:
            assert category in HAZARD_TAXONOMY, f"Missing spec for {category}"

    def test_taxonomy_size(self) -> None:
        assert len(HAZARD_TAXONOMY) == 20

    def test_all_specs_have_required_fields(self) -> None:
        for category, spec in HAZARD_TAXONOMY.items():
            assert spec.category == category
            assert spec.name
            assert spec.description
            assert len(spec.applicable_modalities) > 0
            assert isinstance(spec.severity, Severity)

    def test_hazard_spec_is_frozen(self) -> None:
        spec = HAZARD_TAXONOMY[HazardCategory.VIOLENT_CRIME]
        import dataclasses

        assert dataclasses.is_dataclass(spec)
        # frozen=True means we can't modify
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            spec.name = "changed"  # type: ignore[misc]


class TestGetHazardsForModality:
    def test_text_hazards_include_all_mlcommons(self) -> None:
        text_hazards = get_hazards_for_modality(Modality.TEXT)
        # All MLCommons categories should apply to text
        mlcommons = get_mlcommons_hazards()
        text_codes = {h.category for h in text_hazards}
        for spec in mlcommons:
            if Modality.TEXT in spec.applicable_modalities:
                assert spec.category in text_codes

    def test_audio_hazards_include_voice_impersonation(self) -> None:
        audio_hazards = get_hazards_for_modality(Modality.AUDIO)
        codes = {h.category for h in audio_hazards}
        assert HazardCategory.VOICE_IMPERSONATION in codes

    def test_image_hazards_include_deepfake(self) -> None:
        image_hazards = get_hazards_for_modality(Modality.IMAGE)
        codes = {h.category for h in image_hazards}
        assert HazardCategory.DEEPFAKE_GENERATION in codes

    def test_every_modality_has_hazards(self) -> None:
        for modality in Modality:
            hazards = get_hazards_for_modality(modality)
            assert len(hazards) > 0, f"No hazards for {modality}"


class TestGetMLCommonsHazards:
    def test_returns_12_hazards(self) -> None:
        mlcommons = get_mlcommons_hazards()
        assert len(mlcommons) == 12

    def test_all_aligned(self) -> None:
        for spec in get_mlcommons_hazards():
            assert spec.mlcommons_aligned is True
            assert spec.category.value.startswith("S")


class TestGetCustomHazards:
    def test_returns_8_hazards(self) -> None:
        custom = get_custom_hazards()
        assert len(custom) == 8

    def test_none_aligned(self) -> None:
        for spec in get_custom_hazards():
            assert spec.mlcommons_aligned is False
            assert spec.category.value.startswith("X")
