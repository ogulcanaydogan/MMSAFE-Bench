"""Provider registry for capability-aware routing."""

from __future__ import annotations

from typing import Any

from mmsafe.taxonomy.categories import Modality

# Static registry of known providers and their capabilities.
# Used by the CLI and config validation. Actual provider instances
# are constructed at runtime based on available API keys.
PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "modalities": (Modality.TEXT, Modality.IMAGE),
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-5.2", "dall-e-3"],
        "class": "mmsafe.providers.openai.OpenAIProvider",
    },
    "anthropic": {
        "modalities": (Modality.TEXT,),
        "models": [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
        ],
        "class": "mmsafe.providers.anthropic.AnthropicProvider",
    },
    "google": {
        "modalities": (Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
        "models": ["gemini-2.5-flash", "gemini-3-pro", "veo-3.1", "imagen-3"],
        "class": "mmsafe.providers.google.GoogleProvider",
    },
    "replicate": {
        "modalities": (Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
        "models": [
            "meta/meta-llama-3.1-405b-instruct",
            "black-forest-labs/flux-1.1-pro",
            "kwaivgi/kling-v2.6",
            "resemble-ai/chatterbox",
            "suno-ai/bark",
        ],
        "class": "mmsafe.providers.replicate.ReplicateProvider",
    },
    "elevenlabs": {
        "modalities": (Modality.AUDIO,),
        "models": ["eleven_multilingual_v2", "eleven_turbo_v2.5"],
        "class": "mmsafe.providers.elevenlabs.ElevenLabsProvider",
    },
    "local_vllm": {
        "modalities": (Modality.TEXT,),
        "models": ["any-hf-model"],
        "class": "mmsafe.providers.local_vllm.VLLMProvider",
    },
    "local_ollama": {
        "modalities": (Modality.TEXT,),
        "models": ["llama3.1:8b", "mistral:7b", "any-ollama-model"],
        "class": "mmsafe.providers.local_ollama.OllamaProvider",
    },
    "stub": {
        "modalities": (Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO),
        "models": ["stub-text-v1", "stub-image-v1", "stub-video-v1", "stub-audio-v1"],
        "class": "mmsafe.providers.stub.StubProvider",
    },
}


def get_provider_class(provider_name: str) -> type:
    """Dynamically import and return the provider class."""
    if provider_name not in PROVIDER_REGISTRY:
        msg = (
            f"Unknown provider: {provider_name}. "
            f"Available: {', '.join(PROVIDER_REGISTRY.keys())}"
        )
        raise ValueError(msg)

    class_path = PROVIDER_REGISTRY[provider_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def get_providers_for_modality(modality: Modality) -> list[str]:
    """Return provider names that support the given modality."""
    return [
        name
        for name, info in PROVIDER_REGISTRY.items()
        if modality in info["modalities"]
    ]
