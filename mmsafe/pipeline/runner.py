"""Main evaluation runner — orchestrates the full pipeline."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mmsafe._internal.hardware import (
    concurrency_cap,
    resolve_hardware_profile,
    timeout_floor_seconds,
)
from mmsafe._internal.hashing import short_id
from mmsafe._internal.logging import get_logger, setup_logging
from mmsafe.attacks.registry import get_attack_class
from mmsafe.config.settings import Settings
from mmsafe.judges.composite_judge import CompositeJudge
from mmsafe.judges.registry import get_judge_class
from mmsafe.metrics.aggregator import generate_summary
from mmsafe.pipeline.checkpointing import load_checkpoint, save_checkpoint
from mmsafe.pipeline.executor import PipelineExecutor
from mmsafe.pipeline.result_types import EvalRun, EvalSample
from mmsafe.providers.base import GenerationRequest, ProviderStatus
from mmsafe.providers.registry import get_provider_class
from mmsafe.taxonomy.categories import HazardCategory, Modality

if TYPE_CHECKING:
    from mmsafe.attacks.base import AttackStrategy
    from mmsafe.config.models import EvalConfig
    from mmsafe.judges.base import SafetyJudge
    from mmsafe.providers.base import ModelProvider

log = get_logger("pipeline.runner")
console = Console()


class EvalRunner:
    """Orchestrate the full evaluation pipeline.

    Flow: datasets → attacks → providers → judges → EvalRun
    """

    def __init__(self, config: EvalConfig, resume_from: Path | None = None) -> None:
        self._config = config
        self._resume_from = resume_from
        self._settings = Settings()
        self._provider_init_errors: dict[str, str] = {}

    async def execute(self) -> EvalRun:
        """Run the full evaluation pipeline."""
        setup_logging(self._settings.mmsafe_log_level)

        run_id = f"eval-{uuid.uuid4().hex[:12]}"
        run = EvalRun(run_id=run_id, config_name=self._config.name)
        profile = resolve_hardware_profile(self._config.execution.profile)

        effective_concurrency = self._config.execution.concurrency
        effective_timeout_seconds = self._config.execution.timeout_seconds
        if self._config.execution.auto_tune:
            effective_concurrency = min(
                self._config.execution.concurrency,
                concurrency_cap(profile.profile),
            )
            effective_timeout_seconds = max(
                self._config.execution.timeout_seconds,
                timeout_floor_seconds(profile.profile),
            )

        log.info(
            "Execution profile resolved to %s (reason=%s, concurrency=%d, timeout=%ds)",
            profile.profile,
            profile.reason,
            effective_concurrency,
            effective_timeout_seconds,
        )
        if profile.gpu_names:
            log.info("Visible GPUs: %s", ", ".join(profile.gpu_names))

        # Load checkpoint if resuming
        loaded_completed_ids: set[str] = set()
        if self._resume_from:
            loaded_completed_ids = load_checkpoint(self._resume_from)
        completed_ids = set(loaded_completed_ids)

        # Build components
        providers = await self._build_providers(
            strict=self._config.execution.strict_provider_init
        )
        attacks = self._build_attacks()
        judge = self._build_judge()
        await judge.initialize()

        # Edge simulation
        edge_sim = None
        if self._config.edge.enabled:
            from mmsafe.edge.simulator import EdgeSimulator

            edge_sim = EdgeSimulator(profile_name=self._config.edge.profile)
            log.info("Edge simulation enabled: profile=%s", self._config.edge.profile)

        executor = PipelineExecutor(
            providers=providers,
            judges=[judge],
            max_concurrency=effective_concurrency,
            timeout_seconds=effective_timeout_seconds,
            retry_attempts=self._config.execution.retry_attempts,
        )

        # Load datasets
        prompts = self._load_prompts()
        unavailable_providers = {
            model.provider
            for model in self._config.models
            if model.provider not in providers
        }
        if unavailable_providers:
            log.warning(
                "Skipping unavailable providers: %s",
                ", ".join(sorted(unavailable_providers)),
            )

        total = len(prompts) * len(attacks) * len(self._config.models)
        log.info("Starting evaluation: %d total samples", total)

        sample_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=total)

            for prompt_data in prompts:
                prompt_text = str(prompt_data.get("prompt", ""))
                prompt_modality = str(prompt_data.get("modality", "text"))
                prompt_id = str(prompt_data.get("id", "unknown"))
                prompt_category = str(prompt_data.get("category", "S1"))

                for attack in attacks:
                    # Apply attack transformation
                    request = GenerationRequest(
                        prompt=prompt_text,
                        modality=Modality(prompt_modality),
                        model="",  # Will be set per model
                    )
                    variants = attack.transform(request)

                    for model_spec in self._config.models:
                        if model_spec.provider not in providers:
                            progress.advance(task)
                            continue

                        if prompt_modality not in model_spec.modalities:
                            progress.advance(task)
                            continue

                        for mutated_req, attack_meta in variants:
                            mutated_req = replace(mutated_req, model=model_spec.model)

                            sample_id = short_id(
                                f"{prompt_id}-{attack_meta.strategy_name}"
                                f"-{attack_meta.variant}-{model_spec.model}"
                            )

                            if sample_id in completed_ids:
                                progress.advance(task)
                                continue

                            # Generate
                            response = await executor.generate(
                                model_spec.provider, mutated_req
                            )

                            # Edge constraint check
                            if edge_sim is not None:
                                edge_result = edge_sim.evaluate_response(response)
                                if not edge_result.passed:
                                    log.debug(
                                        "Edge violation for %s: %s",
                                        model_spec.model,
                                        [v.description for v in edge_result.violations],
                                    )

                            # Judge
                            verdict = await executor.judge(mutated_req, response)

                            # Determine outcomes
                            model_refused = (
                                response.refusal_detected
                                or response.status == ProviderStatus.REFUSED
                            )
                            attack_succeeded = not verdict.is_safe and not model_refused

                            category = HazardCategory(prompt_category)

                            sample = EvalSample(
                                sample_id=sample_id,
                                prompt_id=prompt_id,
                                modality=Modality(prompt_modality),
                                hazard_category=category,
                                original_prompt=prompt_text,
                                attack_metadata=attack_meta,
                                request=mutated_req,
                                response=response,
                                verdict=verdict,
                                attack_succeeded=attack_succeeded,
                                model_refused=model_refused,
                            )
                            run.samples.append(sample)

                            sample_count += 1
                            if sample_count % self._config.execution.checkpoint_interval == 0:
                                checkpoint_dir = Path(self._config.output.directory) / "checkpoints"
                                save_checkpoint(
                                    run,
                                    checkpoint_dir,
                                    base_completed_ids=loaded_completed_ids,
                                )

                        progress.advance(task)

        run.completed_at = datetime.now(UTC)
        # Sanitize error messages to avoid leaking API keys or credentials
        sanitized_errors = {
            provider: error.split(":")[0] if "key" in error.lower() else error
            for provider, error in self._provider_init_errors.items()
        }
        run.metadata = {
            "total_samples": len(run.samples),
            "models": [m.model for m in self._config.models],
            "attacks": [a.name for a in self._config.attacks],
            "execution_profile": profile.profile,
            "execution_profile_reason": profile.reason,
            "effective_concurrency": effective_concurrency,
            "effective_timeout_seconds": effective_timeout_seconds,
            "provider_init_errors": sanitized_errors,
            "unavailable_providers": sorted(unavailable_providers),
        }
        if edge_sim is not None:
            run.metadata["edge_simulation"] = edge_sim.get_summary()

        # Persist a final cumulative-safe checkpoint for robust resume behavior.
        checkpoint_dir = Path(self._config.output.directory) / "checkpoints"
        save_checkpoint(
            run,
            checkpoint_dir,
            base_completed_ids=loaded_completed_ids,
        )

        # Generate summary metrics
        summary = generate_summary(run)

        # Save results with summary
        output_dir = Path(self._config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"{run_id}_results.json"
        output_data = {**run.to_dict(), "summary": summary}
        results_path.write_text(json.dumps(output_data, indent=2, default=str))
        log.info("Results saved to %s", results_path)

        # Shutdown
        await judge.shutdown()
        for provider in providers.values():
            await provider.shutdown()

        return run

    async def _build_providers(self, strict: bool = False) -> dict[str, ModelProvider]:
        """Build and initialize provider instances."""
        providers: dict[str, ModelProvider] = {}
        seen: set[str] = set()

        for model_spec in self._config.models:
            if model_spec.provider in seen:
                continue
            seen.add(model_spec.provider)

            try:
                cls = get_provider_class(model_spec.provider)
            except Exception as exc:
                message = f"class resolution failed: {exc}"
                self._provider_init_errors[model_spec.provider] = message
                if strict:
                    raise RuntimeError(
                        f"Failed to resolve provider '{model_spec.provider}': {exc}"
                    ) from exc
                log.warning("Skipping provider '%s': %s", model_spec.provider, message)
                continue

            # Inject API keys / config based on provider
            kwargs = self._provider_kwargs(model_spec.provider)

            try:
                provider = cls(**kwargs)
                await provider.initialize()
                providers[model_spec.provider] = provider
            except Exception as exc:
                message = f"initialization failed: {exc}"
                self._provider_init_errors[model_spec.provider] = message
                if strict:
                    raise RuntimeError(
                        f"Failed to initialize provider '{model_spec.provider}': {exc}"
                    ) from exc
                log.warning("Skipping provider '%s': %s", model_spec.provider, message)

        return providers

    def _provider_kwargs(self, provider_name: str) -> dict[str, Any]:
        """Return constructor kwargs for the given provider name.

        Uses a dispatch mapping instead of a long if/elif chain for clarity
        and maintainability.
        """
        dispatch: dict[str, dict[str, Any]] = {
            "openai": {"api_key": self._settings.openai_api_key},
            "anthropic": {"api_key": self._settings.anthropic_api_key},
            "google": {"api_key": self._settings.google_api_key},
            "replicate": {"api_token": self._settings.replicate_api_token},
            "elevenlabs": {"api_key": self._settings.elevenlabs_api_key},
            "local_vllm": {
                "base_url": self._settings.vllm_base_url,
                "api_key": self._settings.vllm_api_key,
            },
            "local_ollama": {"base_url": self._settings.ollama_base_url},
            "stub": {},
        }
        return dict(dispatch.get(provider_name, {}))

    def _build_attacks(self) -> list[AttackStrategy]:
        """Build attack strategy instances from config."""
        attacks: list[AttackStrategy] = []
        for attack_spec in self._config.attacks:
            cls = get_attack_class(attack_spec.name)
            kwargs: dict[str, object] = {}
            if attack_spec.variants:
                kwargs["variants"] = attack_spec.variants
            if attack_spec.max_turns is not None:
                kwargs["max_turns"] = attack_spec.max_turns
            if attack_spec.escalation:
                kwargs["escalation"] = attack_spec.escalation
            if attack_spec.parameters:
                kwargs.update(attack_spec.parameters)

            try:
                attacks.append(cls(**kwargs))
            except TypeError:
                attacks.append(cls())
        return attacks

    def _build_judge(self) -> SafetyJudge:
        """Build judge from config. Uses composite if multiple judges."""
        if len(self._config.judges) == 1:
            spec = self._config.judges[0]
            cls = get_judge_class(spec.name)
            kwargs: dict[str, object] = {}
            if spec.model:
                kwargs["model"] = spec.model
            if spec.provider:
                kwargs["provider"] = spec.provider
            try:
                return cast("SafetyJudge", cls(**kwargs))
            except TypeError:
                return cast("SafetyJudge", cls())

        composite = CompositeJudge()
        for spec in self._config.judges:
            cls = get_judge_class(spec.name)
            kwargs_j: dict[str, object] = {}
            if spec.model:
                kwargs_j["model"] = spec.model
            if spec.provider:
                kwargs_j["provider"] = spec.provider
            try:
                judge_instance = cast("SafetyJudge", cls(**kwargs_j))
            except TypeError:
                judge_instance = cast("SafetyJudge", cls())
            composite.add_judge(judge_instance, spec.weight)
        return composite

    def _load_prompts(self) -> list[dict[str, object]]:
        """Load prompts from configured datasets."""
        prompts: list[dict[str, object]] = []

        for dataset_spec in self._config.datasets:
            dataset_path = Path(dataset_spec.path)
            if not dataset_path.exists():
                log.warning("Dataset not found: %s, using placeholder", dataset_path)
                # Generate placeholder prompts for testing
                prompts.append({
                    "id": f"placeholder-{dataset_spec.modality}-001",
                    "prompt": f"Test prompt for {dataset_spec.modality} evaluation",
                    "modality": dataset_spec.modality,
                    "category": "S1",
                    "is_benign": str(dataset_spec.is_benign).lower(),
                })
                continue

            with open(dataset_path) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    entry.setdefault("modality", dataset_spec.modality)
                    entry.setdefault("is_benign", str(dataset_spec.is_benign).lower())
                    prompts.append(entry)

                    if (
                        dataset_spec.max_samples
                        and i + 1 >= dataset_spec.max_samples
                    ):
                        break

        return prompts
