"""Safety taxonomy: hazard categories, modalities, and severity levels."""

from mmsafe.taxonomy.categories import (
    ALL_MODALITIES,
    GENERATION_MODALITIES,
    HAZARD_TAXONOMY,
    VISUAL_MODALITIES,
    HazardCategory,
    HazardSpec,
    Modality,
    Severity,
    get_custom_hazards,
    get_hazards_for_modality,
    get_mlcommons_hazards,
)

__all__ = [
    "ALL_MODALITIES",
    "GENERATION_MODALITIES",
    "HAZARD_TAXONOMY",
    "VISUAL_MODALITIES",
    "HazardCategory",
    "HazardSpec",
    "Modality",
    "Severity",
    "get_custom_hazards",
    "get_hazards_for_modality",
    "get_mlcommons_hazards",
]
