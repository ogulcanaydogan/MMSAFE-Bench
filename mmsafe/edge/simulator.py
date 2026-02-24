"""Edge deployment constraint simulation and enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmsafe._internal.logging import get_logger
from mmsafe.edge.constraints import DeploymentConstraints
from mmsafe.edge.profiles import get_profile
from mmsafe.providers.base import GenerationResponse

log = get_logger("edge.simulator")


@dataclass
class ConstraintViolation:
    """Record of a constraint violation during simulation."""

    constraint_type: str  # "memory", "latency", "throughput", "concurrency"
    description: str
    observed_value: float
    limit_value: float


@dataclass
class SimulationResult:
    """Result of applying edge constraints to an evaluation sample."""

    passed: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    simulated_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EdgeSimulator:
    """Enforces hardware constraints during evaluation to simulate edge deployment."""

    def __init__(
        self,
        profile_name: str | None = None,
        constraints: DeploymentConstraints | None = None,
    ) -> None:
        if constraints is not None:
            self._constraints = constraints
        elif profile_name is not None:
            self._constraints = get_profile(profile_name)
        else:
            msg = "Provide either profile_name or constraints"
            raise ValueError(msg)

        self._total_requests = 0
        self._active_requests = 0
        self._violations: list[ConstraintViolation] = []
        log.info("Edge simulator initialized with profile: %s", self._constraints.name)

    @property
    def constraints(self) -> DeploymentConstraints:
        return self._constraints

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def check_model_fit(
        self, model_params_b: float, quantization: str = "fp16",
    ) -> SimulationResult:
        """Check if a model fits within the device's memory constraints."""
        violations = []

        if not self._constraints.can_fit_model(model_params_b, quantization):
            violations.append(
                ConstraintViolation(
                    constraint_type="memory",
                    description=f"Model ({model_params_b}B params, {quantization}) exceeds memory",
                    observed_value=model_params_b,
                    limit_value=self._constraints.resources.max_model_params_b,
                )
            )

        if self._constraints.quantization_required and quantization in ("fp32", "fp16", "bf16"):
            violations.append(
                ConstraintViolation(
                    constraint_type="quantization",
                    description=f"Device requires quantization, got {quantization}",
                    observed_value=0,
                    limit_value=0,
                )
            )

        if quantization not in self._constraints.supported_quantizations:
            violations.append(
                ConstraintViolation(
                    constraint_type="quantization",
                    description=f"Quantization {quantization} not supported by device",
                    observed_value=0,
                    limit_value=0,
                )
            )

        self._violations.extend(violations)
        return SimulationResult(passed=len(violations) == 0, violations=violations)

    def evaluate_response(self, response: GenerationResponse) -> SimulationResult:
        """Evaluate a generation response against edge constraints."""
        violations = []

        # Check latency
        if response.latency_ms > 0:
            latency_ok, latency_violations = self._constraints.check_latency(
                first_token_ms=response.latency_ms * 0.3,  # Estimate TTFT
                total_ms=response.latency_ms,
            )
            if not latency_ok:
                for desc in latency_violations:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="latency",
                            description=desc,
                            observed_value=response.latency_ms,
                            limit_value=self._constraints.latency.max_total_latency_ms,
                        )
                    )

        # Simulate edge latency scaling
        # Edge devices are typically slower; scale by compute ratio
        simulated_latency = response.latency_ms * self._compute_scaling_factor()

        self._violations.extend(violations)
        self._total_requests += 1

        return SimulationResult(
            passed=len(violations) == 0,
            violations=violations,
            simulated_latency_ms=simulated_latency,
            metadata={
                "device": self._constraints.name,
                "scaling_factor": self._compute_scaling_factor(),
            },
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all constraint violations."""
        violation_types: dict[str, int] = {}
        for v in self._violations:
            violation_types[v.constraint_type] = violation_types.get(v.constraint_type, 0) + 1

        return {
            "device_profile": self._constraints.name,
            "total_requests": self._total_requests,
            "total_violations": len(self._violations),
            "violation_breakdown": violation_types,
            "constraints": {
                "max_memory_gb": self._constraints.resources.max_memory_gb,
                "max_tokens_per_second": self._constraints.resources.max_tokens_per_second,
                "max_concurrent_requests": self._constraints.resources.max_concurrent_requests,
                "max_total_latency_ms": self._constraints.latency.max_total_latency_ms,
            },
        }

    def _compute_scaling_factor(self) -> float:
        """Compute latency scaling factor relative to a baseline A100."""
        baseline_tflops = 312.0  # A100 FP16 TFLOPS
        device_tflops = self._constraints.resources.max_compute_tflops
        if device_tflops <= 0:
            return 10.0
        return baseline_tflops / device_tflops
