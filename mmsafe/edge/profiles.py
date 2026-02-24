"""Pre-defined device profiles for edge deployment simulation."""

from __future__ import annotations

from mmsafe.edge.constraints import (
    DeploymentConstraints,
    LatencyConstraints,
    ResourceConstraints,
)

# NVIDIA DGX Spark (Grace Blackwell desktop AI)
DGX_SPARK = DeploymentConstraints(
    name="dgx-spark",
    resources=ResourceConstraints(
        max_memory_gb=128,
        max_compute_tflops=1000,
        max_tokens_per_second=200,
        max_batch_size=8,
        max_model_params_b=200,
        max_concurrent_requests=4,
    ),
    latency=LatencyConstraints(
        max_first_token_ms=500,
        max_inter_token_ms=50,
        max_total_latency_ms=30000,
    ),
    quantization_required=False,
    supported_quantizations=("fp16", "bf16", "int8", "int4"),
)

# NVIDIA Jetson AGX Orin (edge AI module)
JETSON_AGX_ORIN = DeploymentConstraints(
    name="jetson-agx-orin",
    resources=ResourceConstraints(
        max_memory_gb=64,
        max_compute_tflops=275,
        max_tokens_per_second=50,
        max_batch_size=2,
        max_model_params_b=30,
        max_concurrent_requests=2,
    ),
    latency=LatencyConstraints(
        max_first_token_ms=2000,
        max_inter_token_ms=100,
        max_total_latency_ms=60000,
    ),
    quantization_required=True,
    supported_quantizations=("fp16", "int8", "int4"),
)

# Raspberry Pi 5 (minimal edge compute)
RASPBERRY_PI_5 = DeploymentConstraints(
    name="raspberry-pi-5",
    resources=ResourceConstraints(
        max_memory_gb=8,
        max_compute_tflops=0.5,
        max_tokens_per_second=5,
        max_batch_size=1,
        max_model_params_b=3,
        max_concurrent_requests=1,
    ),
    latency=LatencyConstraints(
        max_first_token_ms=10000,
        max_inter_token_ms=500,
        max_total_latency_ms=120000,
    ),
    quantization_required=True,
    supported_quantizations=("int4", "gptq", "awq"),
)

# Apple Mac Studio M4 Ultra (prosumer desktop)
MAC_STUDIO_M4_ULTRA = DeploymentConstraints(
    name="mac-studio-m4-ultra",
    resources=ResourceConstraints(
        max_memory_gb=192,
        max_compute_tflops=600,
        max_tokens_per_second=150,
        max_batch_size=4,
        max_model_params_b=100,
        max_concurrent_requests=4,
    ),
    latency=LatencyConstraints(
        max_first_token_ms=800,
        max_inter_token_ms=40,
        max_total_latency_ms=30000,
    ),
    quantization_required=False,
    supported_quantizations=("fp16", "bf16", "int8", "int4"),
)

# Single NVIDIA V100 (cloud GPU instance)
V100_SINGLE = DeploymentConstraints(
    name="v100-single",
    resources=ResourceConstraints(
        max_memory_gb=32,
        max_compute_tflops=125,
        max_tokens_per_second=80,
        max_batch_size=4,
        max_model_params_b=15,
        max_concurrent_requests=4,
    ),
    latency=LatencyConstraints(
        max_first_token_ms=1000,
        max_inter_token_ms=60,
        max_total_latency_ms=30000,
    ),
    quantization_required=True,
    supported_quantizations=("fp16", "int8", "int4", "gptq"),
)

# Registry of all profiles
DEVICE_PROFILES: dict[str, DeploymentConstraints] = {
    "dgx-spark": DGX_SPARK,
    "jetson-agx-orin": JETSON_AGX_ORIN,
    "raspberry-pi-5": RASPBERRY_PI_5,
    "mac-studio-m4-ultra": MAC_STUDIO_M4_ULTRA,
    "v100-single": V100_SINGLE,
}


def get_profile(name: str) -> DeploymentConstraints:
    """Get a device profile by name."""
    if name not in DEVICE_PROFILES:
        available = ", ".join(DEVICE_PROFILES.keys())
        msg = f"Unknown device profile: {name}. Available: {available}"
        raise ValueError(msg)
    return DEVICE_PROFILES[name]
