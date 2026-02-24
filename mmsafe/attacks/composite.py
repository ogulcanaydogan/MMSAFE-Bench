"""Composite attack that chains multiple strategies together."""

from __future__ import annotations

from mmsafe.attacks.base import AttackCapabilities, AttackMetadata, AttackStrategy
from mmsafe.providers.base import GenerationRequest
from mmsafe.taxonomy.categories import ALL_MODALITIES


class CompositeAttack(AttackStrategy):
    """Chain multiple attack strategies together sequentially.

    Each strategy's output is fed as input to the next strategy,
    creating compound attacks that combine multiple evasion techniques.
    """

    def __init__(self, strategies: list[AttackStrategy] | None = None) -> None:
        self._strategies = strategies or []

    @property
    def name(self) -> str:
        return "composite"

    def capabilities(self) -> AttackCapabilities:
        return AttackCapabilities(
            name="composite",
            description="Chain multiple attack strategies together",
            applicable_modalities=ALL_MODALITIES,
        )

    def transform(
        self, request: GenerationRequest
    ) -> list[tuple[GenerationRequest, AttackMetadata]]:
        if not self._strategies:
            metadata = AttackMetadata(
                strategy_name="composite",
                variant="empty_chain",
                original_prompt=request.prompt,
                transformed_prompt=request.prompt,
            )
            return [(request, metadata)]

        # Apply strategies in sequence: first strategy produces variants,
        # then each variant is fed through subsequent strategies
        current_variants: list[tuple[GenerationRequest, list[AttackMetadata]]] = [
            (request, [])
        ]

        for strategy in self._strategies:
            next_variants: list[tuple[GenerationRequest, list[AttackMetadata]]] = []

            for req, prev_metadata in current_variants:
                step_results = strategy.transform(req)
                for mutated_req, step_meta in step_results:
                    next_variants.append((mutated_req, [*prev_metadata, step_meta]))

            current_variants = next_variants

        # Flatten into final results with composite metadata
        results: list[tuple[GenerationRequest, AttackMetadata]] = []
        strategy_names = [s.name for s in self._strategies]

        for mutated_req, chain_metadata in current_variants:
            composite_meta = AttackMetadata(
                strategy_name="composite",
                variant="+".join(strategy_names),
                parameters={
                    "chain": [
                        {"strategy": m.strategy_name, "variant": m.variant}
                        for m in chain_metadata
                    ]
                },
                original_prompt=request.prompt,
                transformed_prompt=mutated_req.prompt,
            )
            results.append((mutated_req, composite_meta))

        return results
