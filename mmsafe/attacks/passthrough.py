"""Passthrough attack — baseline with no transformation."""

from __future__ import annotations

from mmsafe.attacks.base import AttackCapabilities, AttackMetadata, AttackStrategy
from mmsafe.providers.base import GenerationRequest
from mmsafe.taxonomy.categories import ALL_MODALITIES


class PassthroughAttack(AttackStrategy):
    """No-op baseline: returns the original prompt unchanged."""

    @property
    def name(self) -> str:
        return "passthrough"

    def capabilities(self) -> AttackCapabilities:
        return AttackCapabilities(
            name="passthrough",
            description="Baseline — no transformation applied",
            applicable_modalities=ALL_MODALITIES,
        )

    def transform(
        self, request: GenerationRequest
    ) -> list[tuple[GenerationRequest, AttackMetadata]]:
        metadata = AttackMetadata(
            strategy_name="passthrough",
            original_prompt=request.prompt,
            transformed_prompt=request.prompt,
        )
        return [(request, metadata)]
