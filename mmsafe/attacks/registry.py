"""Attack strategy registry."""

from __future__ import annotations

from typing import Any

from mmsafe.taxonomy.categories import ALL_MODALITIES, Modality

# Static registry of known attack strategies.
ATTACK_REGISTRY: dict[str, dict[str, Any]] = {
    "passthrough": {
        "description": "Baseline — no transformation applied",
        "modalities": ALL_MODALITIES,
        "multi_turn": False,
        "class": "mmsafe.attacks.passthrough.PassthroughAttack",
    },
    "jailbreak": {
        "description": "Jailbreak prompt mutations (DAN, AIM, developer mode)",
        "modalities": ALL_MODALITIES,
        "multi_turn": False,
        "class": "mmsafe.attacks.jailbreak.JailbreakAttack",
    },
    "encoding": {
        "description": "Encoding-based obfuscation (Base64, ROT13, Unicode)",
        "modalities": (Modality.TEXT,),
        "multi_turn": False,
        "class": "mmsafe.attacks.encoding.EncodingAttack",
    },
    "role_play": {
        "description": "Persona and role-play injection",
        "modalities": ALL_MODALITIES,
        "multi_turn": False,
        "class": "mmsafe.attacks.role_play.RolePlayAttack",
    },
    "multi_turn": {
        "description": "Multi-turn gradual escalation",
        "modalities": (Modality.TEXT,),
        "multi_turn": True,
        "class": "mmsafe.attacks.multi_turn.MultiTurnAttack",
    },
    "cross_modal": {
        "description": "Cross-modal injection (text-in-image, audio prompt injection)",
        "modalities": (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
        "multi_turn": False,
        "class": "mmsafe.attacks.cross_modal.CrossModalAttack",
    },
    "adversarial_suffix": {
        "description": "GCG-style adversarial suffix attacks",
        "modalities": (Modality.TEXT,),
        "multi_turn": False,
        "class": "mmsafe.attacks.adversarial_suffix.AdversarialSuffixAttack",
    },
    "translation": {
        "description": "Low-resource language translation attacks",
        "modalities": (Modality.TEXT,),
        "multi_turn": False,
        "class": "mmsafe.attacks.translation.TranslationAttack",
    },
    "composite": {
        "description": "Chain multiple attack strategies together",
        "modalities": ALL_MODALITIES,
        "multi_turn": False,
        "class": "mmsafe.attacks.composite.CompositeAttack",
    },
}


def get_attack_class(attack_name: str) -> type:
    """Dynamically import and return the attack strategy class."""
    if attack_name not in ATTACK_REGISTRY:
        msg = (
            f"Unknown attack: {attack_name}. "
            f"Available: {', '.join(ATTACK_REGISTRY.keys())}"
        )
        raise ValueError(msg)

    class_path = ATTACK_REGISTRY[attack_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]
