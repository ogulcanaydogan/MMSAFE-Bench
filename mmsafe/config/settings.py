"""Environment-based settings for API keys and runtime configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables.

    All API keys are optional — only providers with configured keys will be
    available at runtime.
    """

    # Provider API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    replicate_api_token: str = ""
    elevenlabs_api_key: str = ""

    # Local model endpoints
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Logging
    mmsafe_log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}

    def available_providers(self) -> list[str]:
        """Return list of provider names that have API keys configured."""
        providers: list[str] = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.replicate_api_token:
            providers.append("replicate")
        if self.elevenlabs_api_key:
            providers.append("elevenlabs")
        # Local backends are always potentially available
        providers.extend(["local_vllm", "local_ollama", "stub"])
        return providers
