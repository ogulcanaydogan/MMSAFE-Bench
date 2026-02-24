"""Resource constraint definitions for edge deployment simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceConstraints:
    """Hardware resource limits for a simulated deployment target."""

    max_memory_gb: float
    max_compute_tflops: float
    max_tokens_per_second: float
    max_batch_size: int
    max_model_params_b: float
    max_concurrent_requests: int


@dataclass(frozen=True)
class LatencyConstraints:
    """Latency requirements for the deployment target."""

    max_first_token_ms: float
    max_inter_token_ms: float
    max_total_latency_ms: float


@dataclass(frozen=True)
class DeploymentConstraints:
    """Complete set of constraints for an edge deployment target."""

    name: str
    resources: ResourceConstraints
    latency: LatencyConstraints
    quantization_required: bool = False
    supported_quantizations: tuple[str, ...] = ("fp16", "int8", "int4")

    def can_fit_model(self, model_params_b: float, quantization: str = "fp16") -> bool:
        """Check if a model fits within memory constraints."""
        # Rough estimate: params * bytes_per_param
        bytes_per_param = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.5,
            "gptq": 0.5,
            "awq": 0.5,
        }.get(quantization, 2.0)

        model_size_gb = (model_params_b * 1e9 * bytes_per_param) / (1024**3)
        # Need ~20% overhead for KV cache and runtime
        required_gb = model_size_gb * 1.2
        return required_gb <= self.resources.max_memory_gb

    def check_throughput(self, tokens_per_second: float) -> bool:
        """Check if observed throughput meets requirements."""
        return tokens_per_second <= self.resources.max_tokens_per_second

    def check_latency(self, first_token_ms: float, total_ms: float) -> tuple[bool, list[str]]:
        """Check if latency meets requirements. Returns (pass, violations)."""
        violations = []
        if first_token_ms > self.latency.max_first_token_ms:
            violations.append(
                f"First token {first_token_ms:.0f}ms > {self.latency.max_first_token_ms:.0f}ms"
            )
        if total_ms > self.latency.max_total_latency_ms:
            violations.append(
                f"Total latency {total_ms:.0f}ms > {self.latency.max_total_latency_ms:.0f}ms"
            )
        return len(violations) == 0, violations
