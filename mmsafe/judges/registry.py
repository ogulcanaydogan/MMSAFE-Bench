"""Judge registry for safety evaluation."""

from __future__ import annotations

from typing import Any

from mmsafe.taxonomy.categories import ALL_MODALITIES, Modality

JUDGE_REGISTRY: dict[str, dict[str, Any]] = {
    "keyword_judge": {
        "description": "Regex-based refusal and safety keyword detection",
        "modalities": ALL_MODALITIES,
        "requires_model": False,
        "class": "mmsafe.judges.keyword_judge.KeywordJudge",
    },
    "llm_judge": {
        "description": "LLM-as-judge for nuanced safety evaluation",
        "modalities": ALL_MODALITIES,
        "requires_model": True,
        "class": "mmsafe.judges.llm_judge.LLMJudge",
    },
    "toxicity_judge": {
        "description": "Toxicity scoring via keyword heuristics",
        "modalities": (Modality.TEXT, Modality.AUDIO),
        "requires_model": False,
        "class": "mmsafe.judges.toxicity_judge.ToxicityJudge",
    },
    "nsfw_classifier": {
        "description": "NSFW content detection for images and video",
        "modalities": (Modality.IMAGE, Modality.VIDEO),
        "requires_model": False,
        "class": "mmsafe.judges.nsfw_classifier.NSFWClassifierJudge",
    },
    "composite_judge": {
        "description": "Weighted ensemble of multiple judges",
        "modalities": ALL_MODALITIES,
        "requires_model": False,
        "class": "mmsafe.judges.composite_judge.CompositeJudge",
    },
    "human_eval": {
        "description": "Export samples for human annotation",
        "modalities": ALL_MODALITIES,
        "requires_model": False,
        "class": "mmsafe.judges.human_eval.HumanEvalJudge",
    },
}


def get_judge_class(judge_name: str) -> type:
    """Dynamically import and return the judge class."""
    if judge_name not in JUDGE_REGISTRY:
        msg = (
            f"Unknown judge: {judge_name}. "
            f"Available: {', '.join(JUDGE_REGISTRY.keys())}"
        )
        raise ValueError(msg)

    class_path = JUDGE_REGISTRY[judge_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]
