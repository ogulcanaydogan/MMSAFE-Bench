"""Safety hazard taxonomy with MLCommons alignment and multi-modal extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    """Supported generative modalities."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


ALL_MODALITIES = (Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
VISUAL_MODALITIES = (Modality.IMAGE, Modality.VIDEO)
GENERATION_MODALITIES = (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)


class HazardCategory(str, Enum):
    """MLCommons AILuminate v1.0 hazard categories plus multi-modal extensions.

    S1–S12 are aligned with the MLCommons taxonomy.
    X1–X8 are custom extensions for multi-modal generative safety.
    """

    # MLCommons AILuminate v1.0
    VIOLENT_CRIME = "S1"
    NON_VIOLENT_CRIME = "S2"
    SEX_RELATED_CRIME = "S3"
    CHILD_SAFETY = "S4"
    DEFAMATION = "S5"
    SPECIALIZED_ADVICE = "S6"
    PRIVACY = "S7"
    IP_VIOLATION = "S8"
    INDISCRIMINATE_WEAPONS = "S9"
    HATE_SPEECH = "S10"
    SUICIDE_SELF_HARM = "S11"
    SEXUAL_CONTENT = "S12"

    # Multi-modal extensions
    DEEPFAKE_GENERATION = "X1"
    MISINFORMATION_VISUAL = "X2"
    VOICE_IMPERSONATION = "X3"
    NSFW_GENERATION = "X4"
    BIAS_STEREOTYPING = "X5"
    COPYRIGHT_VIOLATION = "X6"
    ELECTION_MANIPULATION = "X7"
    CROSS_MODAL_BYPASS = "X8"


class Severity(int, Enum):
    """Severity scale for safety hazards."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass(frozen=True)
class HazardSpec:
    """Complete specification for a safety hazard."""

    category: HazardCategory
    name: str
    description: str
    applicable_modalities: tuple[Modality, ...]
    severity: Severity
    mlcommons_aligned: bool = True


# ── Full taxonomy registry ──────────────────────────────────────────────

HAZARD_TAXONOMY: dict[HazardCategory, HazardSpec] = {
    HazardCategory.VIOLENT_CRIME: HazardSpec(
        category=HazardCategory.VIOLENT_CRIME,
        name="Violent Crimes",
        description="Content that enables, encourages, or depicts violent criminal acts "
        "including murder, assault, kidnapping, and terrorism.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CATASTROPHIC,
    ),
    HazardCategory.NON_VIOLENT_CRIME: HazardSpec(
        category=HazardCategory.NON_VIOLENT_CRIME,
        name="Non-Violent Crimes",
        description="Content that enables or encourages non-violent criminal acts "
        "including theft, fraud, hacking, and drug trafficking.",
        applicable_modalities=(Modality.TEXT,),
        severity=Severity.HIGH,
    ),
    HazardCategory.SEX_RELATED_CRIME: HazardSpec(
        category=HazardCategory.SEX_RELATED_CRIME,
        name="Sex-Related Crimes",
        description="Content related to sexual assault, trafficking, "
        "or non-consensual sexual acts.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CATASTROPHIC,
    ),
    HazardCategory.CHILD_SAFETY: HazardSpec(
        category=HazardCategory.CHILD_SAFETY,
        name="Child Sexual Exploitation",
        description="Any content depicting, enabling, or encouraging child sexual "
        "exploitation material (CSEM).",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CATASTROPHIC,
    ),
    HazardCategory.DEFAMATION: HazardSpec(
        category=HazardCategory.DEFAMATION,
        name="Defamation",
        description="Content that makes false claims of fact about specific individuals "
        "that could damage their reputation.",
        applicable_modalities=(Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
        severity=Severity.HIGH,
    ),
    HazardCategory.SPECIALIZED_ADVICE: HazardSpec(
        category=HazardCategory.SPECIALIZED_ADVICE,
        name="Specialized Advice",
        description="Providing unqualified medical, legal, or financial advice "
        "that could cause real-world harm.",
        applicable_modalities=(Modality.TEXT, Modality.AUDIO),
        severity=Severity.HIGH,
    ),
    HazardCategory.PRIVACY: HazardSpec(
        category=HazardCategory.PRIVACY,
        name="Privacy Violations",
        description="Content that exposes personal identifying information, "
        "private communications, or enables surveillance.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CRITICAL,
    ),
    HazardCategory.IP_VIOLATION: HazardSpec(
        category=HazardCategory.IP_VIOLATION,
        name="Intellectual Property",
        description="Content that reproduces copyrighted material, trademarks, "
        "or trade secrets without authorization.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.MEDIUM,
    ),
    HazardCategory.INDISCRIMINATE_WEAPONS: HazardSpec(
        category=HazardCategory.INDISCRIMINATE_WEAPONS,
        name="Indiscriminate Weapons",
        description="Content providing instructions for creating explosives, "
        "chemical weapons, biological agents, or nuclear materials.",
        applicable_modalities=(Modality.TEXT,),
        severity=Severity.CATASTROPHIC,
    ),
    HazardCategory.HATE_SPEECH: HazardSpec(
        category=HazardCategory.HATE_SPEECH,
        name="Hate & Discrimination",
        description="Content that expresses hatred, promotes discrimination, "
        "or dehumanizes individuals based on protected characteristics.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CRITICAL,
    ),
    HazardCategory.SUICIDE_SELF_HARM: HazardSpec(
        category=HazardCategory.SUICIDE_SELF_HARM,
        name="Suicide & Self-Harm",
        description="Content that encourages, instructs, or glorifies "
        "suicide or self-harm.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CATASTROPHIC,
    ),
    HazardCategory.SEXUAL_CONTENT: HazardSpec(
        category=HazardCategory.SEXUAL_CONTENT,
        name="Sexual Content",
        description="Explicit sexual content that is not related to criminal acts "
        "but may be inappropriate in certain contexts.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.MEDIUM,
    ),
    # ── Multi-modal extensions ──────────────────────────────────────────
    HazardCategory.DEEPFAKE_GENERATION: HazardSpec(
        category=HazardCategory.DEEPFAKE_GENERATION,
        name="Deepfake Generation",
        description="Realistic synthetic face or voice content that could be used "
        "for impersonation, fraud, or non-consensual intimate imagery.",
        applicable_modalities=(Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
        severity=Severity.CRITICAL,
        mlcommons_aligned=False,
    ),
    HazardCategory.MISINFORMATION_VISUAL: HazardSpec(
        category=HazardCategory.MISINFORMATION_VISUAL,
        name="Visual Misinformation",
        description="Fabricated images, videos, or audio designed to deceive by "
        "depicting events that never occurred or distorting real events.",
        applicable_modalities=(Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
        severity=Severity.CRITICAL,
        mlcommons_aligned=False,
    ),
    HazardCategory.VOICE_IMPERSONATION: HazardSpec(
        category=HazardCategory.VOICE_IMPERSONATION,
        name="Voice Impersonation",
        description="Unauthorized synthesis of specific individuals' voices "
        "for fraud, scams, or social engineering.",
        applicable_modalities=(Modality.AUDIO,),
        severity=Severity.CRITICAL,
        mlcommons_aligned=False,
    ),
    HazardCategory.NSFW_GENERATION: HazardSpec(
        category=HazardCategory.NSFW_GENERATION,
        name="NSFW Generation",
        description="Generation of explicit visual or audio content "
        "that bypasses content safety filters.",
        applicable_modalities=GENERATION_MODALITIES,
        severity=Severity.HIGH,
        mlcommons_aligned=False,
    ),
    HazardCategory.BIAS_STEREOTYPING: HazardSpec(
        category=HazardCategory.BIAS_STEREOTYPING,
        name="Bias & Stereotyping",
        description="Systematic demographic bias in generated content, "
        "reinforcing harmful stereotypes across gender, race, or culture.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.HIGH,
        mlcommons_aligned=False,
    ),
    HazardCategory.COPYRIGHT_VIOLATION: HazardSpec(
        category=HazardCategory.COPYRIGHT_VIOLATION,
        name="Copyright Violation",
        description="Generation of content that closely replicates copyrighted "
        "works including art styles, music, or written text.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.MEDIUM,
        mlcommons_aligned=False,
    ),
    HazardCategory.ELECTION_MANIPULATION: HazardSpec(
        category=HazardCategory.ELECTION_MANIPULATION,
        name="Election Manipulation",
        description="Content designed to manipulate political processes through "
        "fake endorsements, fabricated speeches, or misleading voter information.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.CATASTROPHIC,
        mlcommons_aligned=False,
    ),
    HazardCategory.CROSS_MODAL_BYPASS: HazardSpec(
        category=HazardCategory.CROSS_MODAL_BYPASS,
        name="Cross-Modal Bypass",
        description="Attacks that exploit modality switching to bypass safety filters, "
        "such as encoding harmful text in images or using audio to inject prompts.",
        applicable_modalities=ALL_MODALITIES,
        severity=Severity.HIGH,
        mlcommons_aligned=False,
    ),
}


def get_hazards_for_modality(modality: Modality) -> list[HazardSpec]:
    """Return all hazard specs applicable to the given modality."""
    return [
        spec
        for spec in HAZARD_TAXONOMY.values()
        if modality in spec.applicable_modalities
    ]


def get_mlcommons_hazards() -> list[HazardSpec]:
    """Return only MLCommons-aligned hazard specs (S1–S12)."""
    return [spec for spec in HAZARD_TAXONOMY.values() if spec.mlcommons_aligned]


def get_custom_hazards() -> list[HazardSpec]:
    """Return only custom multi-modal extension hazard specs (X1–X8)."""
    return [spec for spec in HAZARD_TAXONOMY.values() if not spec.mlcommons_aligned]
